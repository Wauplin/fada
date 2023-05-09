from datasets import load_dataset, load_from_disk
import os
import glob
import importlib
import logging
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

import textattack
from textattack import Attacker
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import HuggingFaceDataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from sibyl import *
from fada.utils import *
from fada.extractors import AMRFeatureExtractor

# https://www.gitmemory.com/issue/QData/TextAttack/424/795806095

def load_class(module_class_str):
    parts = module_class_str.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        out = []
        i = 0
        while i < len(text_input_list):
            batch = text_input_list[i : i + self.batch_size]
            encoding = self.tokenizer(batch, padding=True, truncation=True, max_length=250, return_tensors='pt')
            outputs = self.model(encoding['input_ids'].to(self.device), attention_mask=encoding['attention_mask'].to(self.device))
            preds = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu()
            out.append(preds)
            i += self.batch_size
        out = torch.cat(out)
        return out

@hydra.main(version_base=None, config_path="../fada/conf", config_name="config")
def robustness(cfg: DictConfig) -> None:

    log = logging.getLogger(__name__)

    log.info("Starting robustness evaluation...")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Setting up working directories.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.robustness.visible_cuda_devices
        device = torch.device('cuda')
    log.info(f"training on device={device}")

    #############################################################
    ## Search for pretrained models #############################
    #############################################################

    model_paths = glob.glob(cfg.robustness.model_matcher)
    fine_tuned_models = [m.split("\\")[-1] for m in model_paths]

    #############################################################
    ## Prepare training iterations ##############################
    #############################################################

    run_args = []
    for run_num in range(cfg.robustness.num_runs):
        for model in fine_tuned_models:
            run_args.append({
                "run_num":run_num,
                "fine_tuned_model": model,
            })

    log.info(run_args)

    results = []
    if os.path.exists(cfg.robustness.save_path):
        results.extend(pd.read_csv(cfg.robustness.save_path).to_dict("records"))
        start_position = len(results)
    else:
        start_position = 0

    log.info('starting at position {}'.format(start_position))
    for run_arg in run_args[start_position:]:
        
        #############################################################
        ## Initializations ##########################################
        #############################################################
        run_num = run_arg['run_num']
        trained_model = run_arg['fine_tuned_model']

        log.info(pd.DataFrame([run_arg]))

        #############################################################
        ## Dataset Preparation ######################################
        #############################################################

        annotated_dataset_path = os.path.join(cfg.dataset_dir, 
            f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated.test")
        if os.path.exists(annotated_dataset_path):
            log.info(f"Found existing feature annotated test dataset @ {annotated_dataset_path}!")
            dataset = load_from_disk(annotated_dataset_path)
        else:
            log.info(f"Could not find existing test dataset with feature annotations, generating one and saving @ {annotated_dataset_path}!\nThis may take a while...")
            raw_datasets = load_dataset(cfg.dataset.builder_name, 
                                        cfg.dataset.config_name)        
            if 'sst2' in cfg.dataset.config_name:
                raw_datasets.pop("test") # test set is not usable (all labels -1)
            raw_datasets = prepare_splits(raw_datasets)
            raw_datasets = rename_text_columns(raw_datasets)
            raw_datasets = raw_datasets.shuffle(seed=run_num)
            dataset = raw_datasets["test"]
            if cfg.dataset.text_key != "text" and cfg.dataset.text_key in dataset.features.keys():
                dataset = dataset.rename_column(cfg.dataset.text_key, "text")
            feature_extractor = AMRFeatureExtractor(
                amr_save_path=cfg.amr_extractor.amr_save_path,
                max_sent_len=cfg.amr_extractor.max_sent_len, 
                batch_size=cfg.amr_extractor.batch_size)
            features = feature_extractor(dataset["text"])
            dataset = dataset.add_column("features", [f for f in features])
            dataset.save_to_disk(annotated_dataset_path)
        
        log.info(dataset)
         
        #############################################################
        ## Model + Tokenizer ########################################
        #############################################################

        log.info("Loading model...")
        loaded_checkpoint = False
        if os.path.exists(trained_model):
            tokenizer = AutoTokenizer.from_pretrained(trained_model, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(trained_model).to(device)
            loaded_checkpoint = True

        return 

        #############################################################
        ## TextAttacks ##############################################
        #############################################################

        out = {}
        out.update({
            "run_num":              run_num,
            "trained_model":        trained_model,
            "dataset.builder_name": cfg.dataset.builder_name,
            "dataset.config_name":  cfg.dataset.config_name,
            "num_adversaries":      cfg.num_adversaries,
        })

        if loaded_checkpoint:

            mw = CustomModelWrapper(model, tokenizer)
            dataset = HuggingFaceDataset(dataset, shuffle=True)
            attack_args = textattack.AttackArgs(num_examples=cfg.num_advs, disable_stdout=True)

            for recipe in recipes:

                attack = recipe.build(mw)
                attacker = Attacker(attack, dataset, attack_args)
                attack_results = attacker.attack_dataset()

                num_results = 0
                num_failures = 0
                num_successes = 0

                for result in attack_results:                
                    num_results += 1
                    if (type(result) == textattack.attack_results.SuccessfulAttackResult or 
                        type(result) == textattack.attack_results.MaximizedAttackResult):
                        num_successes += 1
                    if type(result) == textattack.attack_results.FailedAttackResult:
                        num_failures += 1

                attack_success = num_successes / num_results
                out['attack_success_' + recipe.__name__] = attack_success

                print("{0} Attack Success: {1:0.2f}".format(recipe.__name__, attack_success))

        results.append(out)

        # save results
        df = pd.DataFrame(results)
        df.to_csv(cfg.robustness.save_path, index=False)


if __name__ == "__main__":
    robustness()