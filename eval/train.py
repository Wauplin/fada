from datasets import (
    load_dataset,
    load_from_disk
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate

import os
import time
import glob
import logging
import torch
import pandas as pd
import random
import hydra
from omegaconf import DictConfig, OmegaConf

from fada.utils import *
from fada.filters import balance_dataset

random.seed(130)
torch.use_deterministic_algorithms(False)

os.environ["WANDB_DISABLED"] = "true"

#############################################################
## Main Loop Functionality ##################################
#############################################################

@hydra.main(version_base=None, config_path="../fada/conf/", config_name="config")
def train(cfg: DictConfig) -> None:

    log = logging.getLogger(__name__)

    log.info("Starting training...")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Setting up working directories.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.train.visible_cuda_devices)
        device = torch.device('cuda')
    log.info(f"training on device={device}")

    #############################################################
    ## Search for datasets ######################################
    #############################################################

    dataset_paths = glob.glob(os.path.join(cfg.dataset_dir, cfg.train.dataset_matcher))
    dataset_paths = [p.replace("\\", "/") for p in dataset_paths] # fix for formatting issues in windows
    dataset_techniques = [p.split("/")[-1] for p in dataset_paths]

    #############################################################
    ## Prepare training iterations ##############################
    #############################################################

    run_args = []
    for run_num in range(cfg.train.num_runs):
        for model in cfg.train.base_models:
            for technique in dataset_techniques:
                run_args.append({
                    "run_num":run_num,
                    "technique":technique,
                    "base_model":model,
                })

    log.info(run_args)

    results = []
    if os.path.exists(cfg.train.save_path):
        results.extend(pd.read_csv(cfg.train.save_path).to_dict("records"))
        start_position = len(results)
    else:
        start_position = 0

    log.info('starting at position {}'.format(start_position))
    for run_arg in run_args[start_position:]:

        #############################################################
        ## Initializations ##########################################
        #############################################################
        run_num    = run_arg['run_num']
        technique  = run_arg['technique']
        base_model = run_arg['base_model']

        log.info(pd.DataFrame([run_arg]))

        #############################################################
        ## Dataset Preparation ######################################
        #############################################################

        log.info("Loading datasets...")
        raw_datasets = load_dataset(cfg.dataset.builder_name, 
                                    cfg.dataset.config_name)
        if 'sst2' in cfg.dataset.config_name:
            raw_datasets.pop("test") # test set is not usable (all labels -1)

        log.info("Preparing datasets splits...")
        raw_datasets = prepare_splits(raw_datasets)
        raw_datasets = rename_text_columns(raw_datasets)
        raw_datasets = remove_unused_columns(raw_datasets)
        raw_datasets = raw_datasets.shuffle(seed=run_num)
        
        num_labels = len(np.unique(raw_datasets["train"]["label"]))
        
        log.info("Loading prepared training dataset...")
        if technique != "orig":
            save_path = os.path.join(cfg.dataset_dir, technique)
            train_dataset = load_from_disk(save_path)
            raw_datasets["train"] = train_dataset

        if len(raw_datasets["validation"]) > cfg.train.max_val_size:
            log.info("Triming down validation dataset...")
            num_per_class = cfg.train.max_val_size // num_labels
            raw_datasets["validation"] = balance_dataset(raw_datasets["validation"], num_per_class)

        log.info(raw_datasets)
        log.info(f"Number of classes: {num_labels}")

        #############################################################
        ## Model + Tokenize #########################################
        #############################################################
        
        log.info("Loading model + tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels).to(device)

        # tokenize datasets
        def preprocess_function(batch):
            return tokenizer(batch[cfg.dataset.text_key], padding=True, truncation=True, max_length=512)

        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

        #############################################################
        ## Metrics ##################################################
        #############################################################

        log.info("Loading evaluation metrics...")
        metrics = evaluate.combine([
            evaluate.load('accuracy'), 
            ConfiguredMetric(evaluate.load('precision'), average='weighted'),
            ConfiguredMetric(evaluate.load('recall'), average='weighted'),
            ConfiguredMetric(evaluate.load('f1'), average='weighted'),
        ])

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            return metrics.compute(predictions=preds, references = labels)

        #############################################################
        ## Callbacks ################################################
        #############################################################

        callbacks = []
        escb = EarlyStoppingCallback(
            early_stopping_patience=cfg.train.early_stopping_patience
        )
        callbacks.append(escb)

        #############################################################
        ## Training  ################################################
        #############################################################

        base_model_rename = base_model.replace("\\", ".")
        output_dir = os.path.join(cfg.train.trained_models_dir, f"{base_model_rename}.{technique}")

        max_steps = (len(tokenized_datasets["train"]) * cfg.train.num_epochs // cfg.train.gradient_accumulation_steps) // cfg.train.train_batch_size
        logging_steps = (max_steps // cfg.train.num_epochs) // cfg.train.logging_steps_per_epoch

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            max_steps=max_steps,
            save_steps=logging_steps,
            save_total_limit=1,
            per_device_train_batch_size=cfg.train.train_batch_size,
            per_device_eval_batch_size=cfg.train.eval_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps, 
            warmup_steps=logging_steps,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            logging_dir='./training_logs',
            logging_steps=logging_steps,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            evaluation_strategy="steps",
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model, 
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,                  
            train_dataset=tokenized_datasets["train"],         
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            callbacks=callbacks
        )

        log.info("Starting trainer...")
        start_time = time.time()
        trainer.train()
        run_time = time.time() - start_time

        # test with ORIG data
        out = trainer.evaluate(tokenized_datasets["test"])
        out.update({
            "run_num":              run_num,
            "technique":            technique,
            "base_model":           base_model,
            "trained_model":        output_dir,
            "dataset.builder_name": cfg.dataset.builder_name,
            "dataset.config_name":  cfg.dataset.config_name,
            "train_size":           len(tokenized_datasets["train"]),
            "valid_size":           len(tokenized_datasets["validation"]),
            "test_size":            len(tokenized_datasets["test"]),
            "run_time":             run_time,
        })
        log.info('Performance of {}\n{}'.format(output_dir, out))
        results.append(out)

        log.info(f"Saving results to {cfg.train.save_path}")
        df = pd.DataFrame(results)
        df.to_csv(cfg.train.save_path, index=False)

        log.info(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)


if __name__ == "__main__":
    train()