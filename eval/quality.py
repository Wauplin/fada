import os
import glob
import logging
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_from_disk

from fada.utils import *
from fada.extractors import (
    PerformanceExtractor,
    AlignmentMetric,
    GrammarMetric,
    FluencyMetric
)
from fada.filters import balance_dataset

torch.use_deterministic_algorithms(False)

#############################################################
## Main Loop Functionality ##################################
#############################################################

@hydra.main(version_base=None, config_path="../fada/conf/", config_name="config")
def quality(cfg: DictConfig):

    #############################################################
    ## Initializations ##########################################
    #############################################################

    log = logging.getLogger(__name__)

    log.info("Starting quality assessment...")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Setting up working directories.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.train.visible_cuda_devices)
        device = torch.device('cuda')
    log.info(f"ealuating quality on device={device}")

    #############################################################
    ## Search for datasets ######################################
    #############################################################

    dataset_paths = glob.glob(os.path.join(cfg.dataset_dir, cfg.quality.dataset_matcher))
    dataset_paths = [p.replace("\\", "/") for p in dataset_paths] # fix for formatting issues in windows
    dataset_paths = [p for p in dataset_paths if "annotated" not in p]
    dataset_paths.sort()
    dataset_techniques = [p.split("/")[-1] for p in dataset_paths]

    #############################################################
    ## Prepare quality eval iterations ##########################
    #############################################################

    run_args = []
    for technique in dataset_techniques:
        run_args.append({
            "technique":technique,
        })
 
    log.info(run_args)

    results = []
    if os.path.exists(cfg.quality.save_path):
        results.extend(pd.read_csv(cfg.quality.save_path).to_dict("records"))
        start_position = len(results)
    else:
        start_position = 0

    log.info('starting at position {}'.format(start_position))
    for run_arg in run_args[start_position:]:

        #############################################################
        ## Initializations ##########################################
        #############################################################
        technique  = run_arg['technique']

        log.info(pd.DataFrame([run_arg]))

        #############################################################
        ## Load dataset #############################################
        #############################################################

        log.info("Loading datasets...")

        log.info("Loading transformed dataset...")
        save_path = os.path.join(cfg.dataset_dir, technique)
        evaluation_dataset = load_from_disk(save_path).shuffle()

        num_labels = len(np.unique(evaluation_dataset["label"]))
        
        if len(evaluation_dataset) > cfg.quality.max_dataset_size:
            log.info("Triming down dataset dataset...")
            num_per_class = cfg.quality.max_dataset_size // num_labels
            evaluation_dataset = balance_dataset(evaluation_dataset, num_per_class)

        log.info(f"evaluation_dataset: {evaluation_dataset}")

        log.info("Loading comparison dataset of same size as the transformed dataset...")
        raw_datasets = load_dataset(cfg.dataset.builder_name, 
                                    cfg.dataset.config_name)
        
        if 'sst2' in cfg.dataset.config_name:
            raw_datasets.pop("test") # test set is not usable (all labels -1)

        log.info("Preparing datasets splits...")
        raw_datasets = prepare_splits(raw_datasets)
        raw_datasets = rename_text_columns(raw_datasets)
        raw_datasets = remove_unused_columns(raw_datasets)
        raw_datasets = raw_datasets.shuffle()
        num_to_compare = len(raw_datasets["validation"]) if len(evaluation_dataset) > len(raw_datasets["validation"]) else len(evaluation_dataset)
        comparison_dataset = raw_datasets["validation"].select(range(num_to_compare))
        log.info(f"comparison_dataset: {comparison_dataset}")

        #############################################################
        ## Model + Tokenizer ########################################
        #############################################################

        log.info(f"Loading quality model + tokenizer with model_id: {cfg.quality.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.quality.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.quality.model_id, num_labels=num_labels).to(device)

        #############################################################
        ## Initializing Extractors ##################################
        #############################################################
        
        log.info("Initializing metric extractors...")
        perf_extractor = PerformanceExtractor(model=model, tokenizer=tokenizer)
        a_metric = AlignmentMetric(
            builder_name=cfg.dataset.builder_name, 
            config_name=cfg.dataset.config_name,
            model_id=cfg.quality.model_id)
        f_metric = FluencyMetric()
        g_metric = GrammarMetric()

        #############################################################
        ## Quality Evaluations ######################################
        #############################################################

        out = {}
        out.update({
            "quality_model_id":      cfg.quality.model_id,
            "dataset.builder_name":  cfg.dataset.builder_name,
            "dataset.config_name":   cfg.dataset.config_name,
            "dataset_size":          len(evaluation_dataset),
            "technique":             technique,
        })

        log.info("Counting duplication frequency...")
        evaluation_duplicate_count = count_duplicates(evaluation_dataset)
        comparison_duplicate_count = count_duplicates(comparison_dataset)
        out["eval_dupes"] = evaluation_duplicate_count
        out["comp_dupes"] = comparison_duplicate_count
        out["dupe_score"] = out["comp_dupes"] / out["eval_dupes"] if out["eval_dupes"] > 0 else 0
        log.info(f"eval_dupes={out['eval_dupes']}")
        log.info(f"comp_dupes={out['comp_dupes']}")
        log.info(f"dupe_score={out['dupe_score']}")

        log.info("Extracting alignment scores...")
        evaluation_dataset, a_scores_eval = a_metric.evaluate(evaluation_dataset)
        comparison_dataset, a_scores_comp = a_metric.evaluate(comparison_dataset)
        a_scores_eval = a_scores_eval.mean()
        a_scores_comp = a_scores_comp.mean()
        out["eval_alignment"] = a_scores_eval
        out["comp_alignment"] = a_scores_comp
        out["alignment_score"] = out["eval_alignment"] / out["comp_alignment"] 
        log.info(f"eval_alignment={out['eval_alignment']}")
        log.info(f"comp_alignment={out['comp_alignment']}")
        log.info(f"alignment_score={out['alignment_score']}")

        log.info("Extracting fluency scores...")
        evaluation_dataset, f_scores_eval = f_metric.evaluate(evaluation_dataset)
        comparison_dataset, f_scores_comp = f_metric.evaluate(comparison_dataset)
        f_scores_eval = f_scores_eval.mean()
        f_scores_comp = f_scores_comp.mean()
        out["eval_fluency"] = f_scores_eval
        out["comp_fluency"] = f_scores_comp
        out["fluency_score"] = out["comp_fluency"] / out["eval_fluency"]
        log.info(f"eval_fluency={out['eval_fluency']}")
        log.info(f"comp_fluency={out['comp_fluency']}")
        log.info(f"fluency_score={out['fluency_score']}")

        log.info("Extracting grammaticality scores...")
        evaluation_dataset, g_scores_eval = g_metric.evaluate(evaluation_dataset)
        comparison_dataset, g_scores_comp = g_metric.evaluate(comparison_dataset)
        g_scores_eval = g_scores_eval.mean()
        g_scores_comp = g_scores_comp.mean()
        out["eval_grammaticality"] = g_scores_eval
        out["comp_grammaticality"] = g_scores_comp
        out["grammaticality_score"] = out["comp_grammaticality"] / out["eval_grammaticality"]
        log.info(f"eval_grammaticality={out['eval_grammaticality']}")
        log.info(f"comp_grammaticality={out['comp_grammaticality']}")
        log.info(f"grammaticality_score={out['grammaticality_score']}")

        log.info("Extracting model perfromance scores...")
        p_scores_eval = perf_extractor.extract_performance(evaluation_dataset)
        p_scores_comp = perf_extractor.extract_performance(comparison_dataset)
        out["eval_gutcheck_accuracy"]   = p_scores_eval["accuracy"]
        out["eval_gutcheck_precision"]  = p_scores_eval["precision"]
        out["eval_gutcheck_recall"]     = p_scores_eval["recall"]
        out["eval_gutcheck_f1"]         = p_scores_eval["f1"]
        out["comp_gutcheck_accuracy"]   = p_scores_comp["accuracy"]
        out["comp_gutcheck_precision"]  = p_scores_comp["precision"]
        out["comp_gutcheck_recall"]     = p_scores_comp["recall"]
        out["comp_gutcheck_f1"]         = p_scores_comp["f1"]
        out["gutcheck_accuracy_score"]  = out["eval_gutcheck_accuracy"]  / out["comp_gutcheck_accuracy"] 
        out["gutcheck_precision_score"] = out["eval_gutcheck_precision"] / out["comp_gutcheck_precision"] 
        out["gutcheck_recall_score"]    = out["eval_gutcheck_recall"]    / out["comp_gutcheck_recall"] 
        out["gutcheck_f1_score"]        = out["eval_gutcheck_f1"]        / out["comp_gutcheck_f1"] 
        log.info(f"eval_gutcheck_accuracy={out['eval_gutcheck_accuracy']}")
        log.info(f"eval_gutcheck_precision={out['eval_gutcheck_precision']}")
        log.info(f"eval_gutcheck_recall={out['eval_gutcheck_recall']}")
        log.info(f"eval_gutcheck_f1={out['eval_gutcheck_f1']}")
        log.info(f"comp_gutcheck_accuracy={out['comp_gutcheck_accuracy']}")
        log.info(f"comp_gutcheck_precision={out['comp_gutcheck_precision']}")
        log.info(f"comp_gutcheck_recall={out['comp_gutcheck_recall']}")
        log.info(f"comp_gutcheck_f1={out['comp_gutcheck_f1']}")
        log.info(f"gutcheck_accuracy_score={out['gutcheck_accuracy_score']}")
        log.info(f"gutcheck_precision_score={out['gutcheck_precision_score']}")
        log.info(f"gutcheck_recall_score={out['gutcheck_recall_score']}")
        log.info(f"gutcheck_f1_score={out['gutcheck_f1_score']}")

        results.append(out)

        log.info(f"Saving results to {cfg.quality.save_path}")
        df = pd.DataFrame(results)
        df.to_csv(cfg.quality.save_path, index=False)

if __name__ == "__main__":
    quality()