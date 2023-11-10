import os
import glob
import logging
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import traceback 

from datasets import load_dataset, load_from_disk

from fada.transform import Transform
from fada.extractors import (
    AMRFeatureExtractor,
    AlignmentMetric,
    FluencyMetric,
    GrammarMetric,
    DocumentSemanticDiversity,
    DocumentDependencyParseDiversity,
    DocumentPartOfSpeechSequenceDiversity,
    MATTRDiversity,
    UniqueBigramsDiversity
)
from fada.augmenter import Augmenter
from fada.utils import (
    load_class,
    policy_heatmap, 
    prepare_splits,
    rename_text_columns,
    remove_unused_columns_from_dataset
)
from fada.filters import balance_dataset

log = logging.getLogger(__name__)
torch.use_deterministic_algorithms(False)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def fada_search(cfg: DictConfig) -> None:

    log.info("Starting fada_search.")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Setting up working directories.")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.amr_extractor.amr_dir, exist_ok=True)
    os.makedirs(cfg.fada.tfim_dir, exist_ok=True)
    os.makedirs(cfg.quality.trial_dir, exist_ok=True)

    log.info("Loading & initializing transforms.")
    transforms = [load_class(t) for t in cfg.transforms]
    transforms = sorted(transforms, key=lambda t: t.__name__)
    transforms = [Transform(t, task_name=cfg.dataset.task_name) for t in transforms]
    num_transforms = len(transforms)
    log.info(f"Running fada search with {num_transforms} to choose from...")

    log.info("Checking to see if fada_search has been run for this dataset before...")
    matrix_paths = glob.glob(os.path.join(cfg.fada.tfim_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada{num_transforms}*"))
    if len(matrix_paths) > 0:
        log.info("Found existing TFIM metadata!")
        if not cfg.fada.force:
            log.info("fada.force=False. Using existing TFIM metadata.")
            return
        else:
            log.info("fada.force=True. Continuing with new fada_search. This will override existing metadata.")
    else:
        log.info("No existing fada search results found. Continuing with fada_search.")

    log.info("Initializing metric extractors.")
    feature_extractor = AMRFeatureExtractor(
        amr_save_path=cfg.amr_extractor.amr_save_path,
        max_sent_len=cfg.amr_extractor.max_sent_len, 
        batch_size=cfg.amr_extractor.batch_size)
    
    # quality metrics
    a_metric = AlignmentMetric(
        builder_name=cfg.dataset.builder_name, 
        config_name=cfg.dataset.config_name,
        model_id=cfg.alignment_extractor.model_id)
    f_metric = FluencyMetric()
    g_metric = GrammarMetric()
    
    # diversity metrics
    d_sem_metric = DocumentSemanticDiversity()
    d_syn_metric = DocumentDependencyParseDiversity()
    d_mor_metric = DocumentPartOfSpeechSequenceDiversity()
    d_mtr_metric = MATTRDiversity()
    d_ubi_metric = UniqueBigramsDiversity()
    
    log.info("Loading dataset...")
    annotated_dataset_path = os.path.join(cfg.dataset_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated")
    if os.path.exists(annotated_dataset_path):
        log.info(f"Found existing feature annotated dataset @ {annotated_dataset_path}!")
        dataset = load_from_disk(annotated_dataset_path)
        features = np.array(dataset["features"])
        if "preds" in dataset.column_names:
            dataset = dataset.remove_columns("preds")
    else:
        log.info(f"Could not find existing dataset with feature annotations, generating one and saving @ {annotated_dataset_path}!\nThis may take a while...")
        raw_datasets = load_dataset(cfg.dataset.builder_name, 
                                    cfg.dataset.config_name)
        raw_datasets = prepare_splits(raw_datasets)
        raw_datasets = rename_text_columns(raw_datasets)
        dataset = raw_datasets["train"]
        if cfg.dataset.text_key != "text" and cfg.dataset.text_key in dataset.features.keys():
            dataset = dataset.rename_column(cfg.dataset.text_key, "text")

        if dataset.num_rows > cfg.dataset.max_size:
            log.info(f"Dataset size is larger than dataset.max_size={cfg.dataset.max_size}")
            if str(cfg.dataset.num_per_class) == "infer":
                num_labels = len(np.unique(dataset['label']))
                num_per_class = int(cfg.dataset.max_size / num_labels)
            else:
                num_per_class = cfg.dataset.num_per_class
            log.info(f"Balancing dataset with num_per_class={num_per_class}")
            dataset = balance_dataset(dataset, num_per_class=num_per_class)
            log.info(dataset)

        features = feature_extractor(dataset["text"])
        dataset = dataset.add_column("features", [f for f in features])
        dataset.save_to_disk(annotated_dataset_path)
    
    dataset = remove_unused_columns_from_dataset(dataset)

    log.info(dataset)

    log.info("Beginning FADA search procedure...")
    num_transforms   = len(cfg.transforms)
    num_features     = len(cfg.features)

    alignment_scores = np.zeros((num_transforms, num_features))
    fluency_scores   = np.zeros((num_transforms, num_features))
    grammar_scores   = np.zeros((num_transforms, num_features))
    sem_div_scores   = np.zeros((num_transforms, num_features))
    syn_div_scores   = np.zeros((num_transforms, num_features))
    mor_div_scores   = np.zeros((num_transforms, num_features))
    mtr_div_scores   = np.zeros((num_transforms, num_features))
    ubi_div_scores   = np.zeros((num_transforms, num_features))
    counts           = np.zeros((num_transforms, num_features))
    changes          = np.zeros((num_transforms, num_features))
    tfim             = np.full((num_transforms, num_features), fill_value=1/num_transforms)

    tfim_difference = np.inf
    convergence_threshold = 1 / (num_transforms + num_features)

    trial_data = []
    i = 0
    while tfim_difference > convergence_threshold:

        # find low coverage (t,f) pairs
        ts, fs   = np.where(changes < cfg.fada.min_coverage)
        tf_pairs = list(zip(ts, fs))

        for t, f in tqdm(tf_pairs):

            f_candidates  = np.where(features[:,f] == 1)[0]

            # feature missing in dataset
            if not f_candidates.size:
                continue

            num_to_sample = cfg.fada.num_to_transform_per_step if len(f_candidates) > cfg.fada.num_to_transform_per_step else len(f_candidates)
            f_indices     = np.random.choice(f_candidates, num_to_sample, replace=False)
            f_dataset     = dataset.select(f_indices)
            f_dataset     = f_dataset.remove_columns(["features"])

            t_prob = np.zeros(num_transforms)
            t_prob[t] = 1
            transform_probabilities = np.array([t_prob for _ in range(f_dataset.num_rows)])
            try:
                augmenter = Augmenter(dataset=f_dataset, 
                            transforms=transforms,  
                            transform_probabilities=transform_probabilities,
                            num_augmentations_per_record=cfg.augment.num_augmentations_per_record,
                            num_transforms_to_apply=cfg.augment.num_transforms_to_apply,
                            batch_size=cfg.augment.batch_size, 
                            keep_originals=False)
                aug_dataset = augmenter.augment()
            except Exception as e:
                traceback.print_exc()
                log.error(e)
                continue

            
            a_start_time = time.time()
            aug_dataset, a_scores = a_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            a_time = time.time() - a_start_time
            f_start_time = time.time()
            aug_dataset, f_scores = f_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            f_time = time.time() - f_start_time
            g_start_time = time.time()
            aug_dataset, g_scores = g_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            g_time = time.time() - g_start_time

            a_score = a_scores.mean()
            f_score = f_scores.mean()
            g_score = g_scores.mean()
            
            sem_start_time = time.time()
            aug_dataset, sem_div  = d_sem_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            sem_time = time.time() - sem_start_time
            syn_start_time = time.time()
            aug_dataset, syn_div  = d_syn_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            syn_time = time.time() - syn_start_time
            mor_start_time = time.time()
            aug_dataset, mor_div  = d_mor_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            mor_time = time.time() - mor_start_time
            mtr_start_time = time.time()
            aug_dataset, mtr_div  = d_mtr_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            mtr_time = time.time() - mtr_start_time
            ubi_start_time = time.time()
            aug_dataset, ubi_div  = d_ubi_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            ubi_time = time.time() - ubi_start_time

            alignment_scores[t,f] = np.clip((alignment_scores[t,f] + a_score) / 2, 0, 2)
            fluency_scores[t,f]   = np.clip((fluency_scores[t,f]   + f_score) / 2, 0, 2)
            grammar_scores[t,f]   = np.clip((grammar_scores[t,f]   + g_score) / 2, 0, 2)

            sem_div_scores[t,f]   = np.clip((sem_div_scores[t,f]   + sem_div) / 2, 0, 2)
            syn_div_scores[t,f]   = np.clip((syn_div_scores[t,f]   + syn_div) / 2, 0, 2)
            mor_div_scores[t,f]   = np.clip((mor_div_scores[t,f]   + mor_div) / 2, 0, 2)
            mtr_div_scores[t,f]   = np.clip((mtr_div_scores[t,f]   + mtr_div) / 2, 0, 2)
            ubi_div_scores[t,f]   = np.clip((ubi_div_scores[t,f]   + ubi_div) / 2, 0, 2)

            counts[t,f]           += f_dataset.num_rows
            changes[t,f]          += np.array(aug_dataset["is_changed"]).sum()

            trial_out = {
                "trial_num": i, 
                "transform": t,
                "feature": f, 
                "builder_name": cfg.dataset.builder_name, 
                "config_name": cfg.dataset.config_name,
                "model_id": cfg.alignment_extractor.model_id,
                "dataset_size": len(f_dataset),
                "alignment_score": a_score,
                "fluency_score": f_score,
                "grammaticality_score": g_score,
                "semantic_diversity": sem_div,
                "syntactic_diversity": syn_div,
                "morphological_diversity": mor_div,
                "mattr_diversity": mtr_div,
                "unique_bigram_diversity": ubi_div,
                "a_time": a_time,
                "f_time": f_time,
                "g_time": g_time,
                "sem_time": sem_time,
                "syn_time": syn_time,
                "mor_time": mor_time,
                "mtr_time": mtr_time,
                "ubi_time": ubi_time,
            }
            log.info(trial_out)
            trial_data.append(trial_out)

        # compute tfim-augment
        aggregated_performance = (cfg.fada.c_a * alignment_scores) + \
                                 (cfg.fada.c_f * fluency_scores) + \
                                 (cfg.fada.c_g * grammar_scores) + \
                                 (cfg.fada.c_div_sem * sem_div_scores) + \
                                 (cfg.fada.c_div_syn * syn_div_scores) + \
                                 (cfg.fada.c_div_mor * mor_div_scores) + \
                                 (cfg.fada.c_div_mtr * mtr_div_scores) + \
                                 (cfg.fada.c_div_ubi * ubi_div_scores)
        
        applicability_rate     = np.nan_to_num(changes / counts, 0)
        new_tfim               = softmax(applicability_rate * aggregated_performance, axis=0)
        tfim_difference        = np.linalg.norm(new_tfim - tfim)
        tfim                   = new_tfim

        log.info(f"tfim_difference: {tfim_difference} (convergence_threshold: {convergence_threshold})")

        policy_heatmap(tfim, transforms, feature_extractor.featurizers)

        log.info("Saving intermediate matrices...")
        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada{num_transforms}"
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.counts-step-{i}"), counts)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.changes-step-{i}"), changes)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.alignment-step-{i}"), alignment_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.fluency-step-{i}"), fluency_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.grammar-step-{i}"), grammar_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.div_sem-step-{i}"), sem_div_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.div_syn-step-{i}"), syn_div_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.div_mor-step-{i}"), mor_div_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.div_mtr-step-{i}"), mtr_div_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.div_ubi-step-{i}"), ubi_div_scores)
        np.save(os.path.join(cfg.fada.tfim_dir, f"{save_name}.tfim-step-{i}"), tfim)

        log.info(f"Saving intermediate trial information for the quality study")
        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada{num_transforms}.trial_data.csv"
        save_path = os.path.join(cfg.quality.trial_dir, save_name)
        df = pd.DataFrame(trial_data)
        df.to_csv(save_path, index=False)

        i += 1
        
        if i > cfg.fada.max_iterations:
            break

    log.info(f"FADA policy generation for {cfg.dataset.builder_name}.{cfg.dataset.config_name} complete!")

if __name__ == "__main__":
    fada_search()
