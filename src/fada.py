import os
import importlib
import logging
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import hydra
from omegaconf import DictConfig

from datasets import load_dataset, load_from_disk

from transform import Transform
from extractors import (
    AMRFeatureExtractor,
    AlignmentMetric,
    FluencyMetric,
    GrammarMetric
)
from augmenter import Augmenter
from utils import policy_heatmap

def load_class(module_class_str):
    parts = module_class_str.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def fada_search(cfg: DictConfig) -> None:

    # setup working directories
    log.info("Setting up working directories.")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.tfim_dir, exist_ok=True)

    # # initialize transforms
    log.info("Loading & initializing transforms.")
    transforms = [load_class(t) for t in cfg.transforms]
    transforms = sorted(transforms, key=lambda t: t.__name__)
    transforms = [Transform(t, task_name=cfg.dataset.task_name) for t in transforms]
    print(transforms)

    # initialize metric extractors
    log.info("Initializing metric extractors.")
    a_metric = AlignmentMetric()
    f_metric = FluencyMetric()
    g_metric = GrammarMetric()

    # load dataset + annotations
    log.info("Loading dataset...")
    annotated_dataset = f"{cfg.dataset_dir}{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated"

    # initialize dataset 
    if os.path.exists(annotated_dataset):
        log.info(f"Found existing feature annotated dataset @ {annotated_dataset}!")
        dataset = load_from_disk(annotated_dataset)
        features = np.array(dataset["features"])
    else:
        log.info(f"Could not find existing dataset with feature annotations, generating one and saving @ {annotated_dataset}!")
        log.info("This may take a while...")
        feature_extractor = AMRFeatureExtractor()
        dataset = load_dataset(cfg.dataset.builder_name, 
                               cfg.dataset.config_name, 
                               split="train")
        if cfg.dataset.text_key != "text":
            dataset = dataset.rename_column(cfg.dataset.text_key, "text")
        features = feature_extractor(dataset["text"])
        dataset = dataset.add_column("features", [f for f in features])
        dataset.save_to_disk(annotated_dataset)
    print(dataset)

    # initialize fadata arrays
    log.info("Beginning FADA search procedure...")
    num_transforms   = len(cfg.transforms)
    num_features     = len(cfg.features)

    alignment_scores = np.zeros((num_transforms, num_features))
    fluency_scores   = np.zeros((num_transforms, num_features))
    grammar_scores   = np.zeros((num_transforms, num_features))
    counts           = np.zeros((num_transforms, num_features))
    changes          = np.zeros((num_transforms, num_features))
    tfim             = np.full((num_transforms, num_features), fill_value=1/num_transforms)

    tfim_difference = np.inf
    convergence_threshold = 1 / (num_transforms + num_features)

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

            t_prob = np.zeros(num_transforms)
            t_prob[t] = 1
            transform_probabilities = np.array([t_prob for _ in range(f_dataset.num_rows)])
            augmenter = Augmenter(dataset=f_dataset, 
                        transforms=transforms,  
                        transform_probabilities=transform_probabilities,
                        num_augmentations_per_record=cfg.fada.num_augmentations_per_record,
                        num_transforms_to_apply=cfg.fada.num_transforms_to_apply,
                        batch_size=cfg.fada.batch_size, 
                        keep_originals=False)
            aug_dataset = augmenter.augment()

            aug_dataset, a_scores = a_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            aug_dataset, f_scores = f_metric.evaluate_before_and_after(f_dataset, aug_dataset)
            aug_dataset, g_scores = g_metric.evaluate_before_and_after(f_dataset, aug_dataset)

            alignment_scores[t,f] = np.clip((alignment_scores[t,f] + a_scores.mean()) / 2, 0, 2)
            fluency_scores[t,f]   = np.clip((fluency_scores[t,f]   + f_scores.mean()) / 2, 0, 2)
            grammar_scores[t,f]   = np.clip((grammar_scores[t,f]   + g_scores.mean()) / 2, 0, 2)
            counts[t,f]           += f_dataset.num_rows
            changes[t,f]          += np.array(aug_dataset["is_changed"]).sum()

        # compute tfim-augment
        aggregated_performance = (cfg.fada.c_a * alignment_scores) + \
                                 (cfg.fada.c_f * fluency_scores) + \
                                 (cfg.fada.c_g * grammar_scores)
        applicability_rate     = np.nan_to_num(changes / counts, 0)
        new_tfim               = softmax(applicability_rate * aggregated_performance, axis=0)
        tfim_difference        = np.linalg.norm(new_tfim - tfim)
        tfim                   = new_tfim

        log.info(f"tfim_difference: {tfim_difference} (convergence_threshold: {convergence_threshold})")

        policy_heatmap(tfim, transforms, feature_extractor.featurizers)

        log.info("Saving intermediate matrices...")
        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada"
        np.save(os.path.join(cfg.tfim_dir, f"{save_name}.counts-step-{i}"), counts)
        np.save(os.path.join(cfg.tfim_dir, f"{save_name}.changes-step-{i}"), changes)
        np.save(os.path.join(cfg.tfim_dir, f"{save_name}.alignment-step-{i}"), alignment_scores)
        np.save(os.path.join(cfg.tfim_dir, f"{save_name}.fluency-step-{i}"), fluency_scores)
        np.save(os.path.join(cfg.tfim_dir, f"{save_name}.grammar-step-{i}"), grammar_scores)
        np.save(os.path.join(cfg.tfim_dir, f"{save_name}.tfim-step-{i}"), tfim)

        i += 1
        
        if i > cfg.fada.max_iterations:
            break
    
    log.info(f"FADA policy generation complete!")

if __name__ == "__main__":
    fada_search()