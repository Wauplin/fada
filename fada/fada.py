import os
import importlib
import glob
import logging
from tqdm import tqdm
import torch
import numpy as np
from scipy.special import softmax
import hydra
from omegaconf import DictConfig

from datasets import load_dataset, load_from_disk

from fada.transform import Transform
from extractors import (
    AMRFeatureExtractor,
    AlignmentMetric,
    FluencyMetric,
    GrammarMetric
)
from fada.augmenter import Augmenter
from fada.utils import (
    policy_heatmap, 
    implement_policy_probabilities,
    prepare_splits,
    rename_text_columns
)
from fada.filters import balance_dataset

def load_class(module_class_str):
    parts = module_class_str.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls

log = logging.getLogger(__name__)
torch.use_deterministic_algorithms(False)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def fada_search(cfg: DictConfig) -> None:

    log.info("Starting fada_search.")

    log.info("Setting up working directories.")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.amr_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.tfim_dir, exist_ok=True)

    log.info("Checking to see if fada_search has been run for this dataset before...")
    matrix_paths = glob.glob(os.path.join(cfg.tfim_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}*"))
    if len(matrix_paths) > 0:
        log.info("Found existing TFIM metadata!")
        if not cfg.fada.force:
            log.info("fada.force=False. Using existing TFIM metadata.")
            return
        else:
            log.info("fada.force=True. Continuing with new fada_search. This will override existing metadata.")
    else:
        log.info("No existing fada search results found. Continuing with fada_search.")

    log.info("Loading & initializing transforms.")
    transforms = [load_class(t) for t in cfg.transforms]
    transforms = sorted(transforms, key=lambda t: t.__name__)
    transforms = [Transform(t, task_name=cfg.dataset.task_name) for t in transforms]

    log.info("Initializing metric extractors.")
    feature_extractor = AMRFeatureExtractor(
        amr_save_path=os.path.join(cfg.amr_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.pkl"),
        max_sent_len=cfg.amr_extractor.max_sent_len, 
        batch_size=cfg.amr_extractor.batch_size)
    a_metric = AlignmentMetric(
        builder_name=cfg.dataset.builder_name, 
        config_name=cfg.dataset.config_name,
        model_id=cfg.alignment_extractor.model_id)
    f_metric = FluencyMetric()
    g_metric = GrammarMetric()

    log.info("Loading dataset...")
    annotated_dataset = f"{cfg.dataset_dir}{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated"
    if os.path.exists(annotated_dataset):
        log.info(f"Found existing feature annotated dataset @ {annotated_dataset}!")
        dataset = load_from_disk(annotated_dataset)
        features = np.array(dataset["features"])
    else:
        log.info(f"Could not find existing dataset with feature annotations, generating one and saving @ {annotated_dataset}!")
        log.info("This may take a while...")
        raw_datasets = load_dataset(cfg.dataset.builder_name, 
                                    cfg.dataset.config_name)
        raw_datasets = prepare_splits(raw_datasets)
        raw_datasets = rename_text_columns(raw_datasets)
        dataset = raw_datasets["train"]
        if cfg.dataset.text_key != "text":
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
        dataset.save_to_disk(annotated_dataset)
    print(dataset)

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
                        num_augmentations_per_record=cfg.augment.num_augmentations_per_record,
                        num_transforms_to_apply=cfg.augment.num_transforms_to_apply,
                        batch_size=cfg.augment.batch_size, 
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
    
    log.info(f"FADA policy generation for {cfg.dataset.builder_name}.{cfg.dataset.config_name} complete!")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def fada_augment(cfg: DictConfig) -> None:

    log.info("Starting fada_augment.")

    log.info("Setting up working directories.")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.tfim_dir, exist_ok=True)

    log.info("Checking to see if fada_augment has been run for this dataset before...")
    matrix_paths = glob.glob(os.path.join(cfg.dataset_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.{cfg.augment.technique}*"))
    if len(matrix_paths) > 0:
        log.info("Found existing augmented dataset!")
        if not cfg.augment.force:
            log.info("augment.force=False. Skipping augmentation...")
            return
        else:
            log.info("augment.force=True. Continuing with new augmentation. This will override the existing dataset(s).")
    else:
        log.info("No existing dataset(s) found. Continuing with fada_augment.")

    log.info("Loading & initializing transforms.")
    transforms = [load_class(t) for t in cfg.transforms]
    transforms = sorted(transforms, key=lambda t: t.__name__)
    transforms = [Transform(t, task_name=cfg.dataset.task_name) for t in transforms]
    
    log.info("Loading dataset...")
    annotated_dataset = f"{cfg.dataset_dir}{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated"
    if os.path.exists(annotated_dataset):
        dataset = load_from_disk(annotated_dataset)
        features = np.array(dataset["features"])
        keep_cols = ['text', 'label', 'idx']
        dataset = dataset.remove_columns([c for c in dataset.features.keys() if c not in keep_cols])
    
    log.info(f"Beginning augmentation for technique={cfg.augment.technique}")

    if "uniform" in cfg.augment.technique:
        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.{cfg.augment.technique}"
        save_path = os.path.join(cfg.dataset_dir, save_name)
        augmenter = Augmenter(dataset=dataset, 
                    transforms=transforms,  
                    transform_probabilities=None,
                    num_augmentations_per_record=cfg.augment.num_augmentations_per_record,
                    num_transforms_to_apply=cfg.augment.num_transforms_to_apply,
                    batch_size=cfg.augment.batch_size, 
                    keep_originals=cfg.augment.keep_originals)
        aug_dataset = augmenter.augment()
        aug_dataset.save_to_disk(save_path)
        torch.cuda.empty_cache()
        log.info(f"{cfg.augment.technique} augmented dataset saved @ {save_path}!")
        
    if "fada" in cfg.augment.technique:

        log.info("Loading final TFIM component matrices.")
        matrix_paths = glob.glob(os.path.join(cfg.tfim_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}*"))
        max_id = int(max([m.split("-")[-1].split(".")[0] for m in matrix_paths]))

        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada"
        counts    = np.load(os.path.join(cfg.tfim_dir, f"{save_name}.counts-step-{max_id}.npy"))
        changes   = np.load(os.path.join(cfg.tfim_dir, f"{save_name}.changes-step-{max_id}.npy"))
        alignment = np.load(os.path.join(cfg.tfim_dir, f"{save_name}.alignment-step-{max_id}.npy"))
        fluency   = np.load(os.path.join(cfg.tfim_dir, f"{save_name}.fluency-step-{max_id}.npy"))
        grammar   = np.load(os.path.join(cfg.tfim_dir, f"{save_name}.grammar-step-{max_id}.npy"))
        tfim      = np.load(os.path.join(cfg.tfim_dir, f"{save_name}.tfim-step-{max_id}.npy"))

        if "fada-sweep" in cfg.augment.technique:

            log.info("Generating new TFIMs representing different linear combinations of alignment, fluency, and grammaticality.")
            for c_a in np.linspace(0.1, 1, 10):
                c_f = c_g = (1 - c_a) / 2
                c_a, c_f, c_g = round(c_a, 2), round(c_f, 2), round(c_g, 2)
                log.info(f"c_a: {c_a}, c_f: {c_f}, c_g: {c_g}")
                
                aggregated_performance = (c_a * alignment) + \
                                        (c_f * fluency) + \
                                        (c_g * grammar)
                applicability_rate = np.nan_to_num(changes / counts, 0)
                tfim = softmax(applicability_rate * aggregated_performance, axis=0)

                policy_probabilities = implement_policy_probabilities(tfim, features)

                save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.{cfg.augment.technique}.a.{c_a}.f.{c_f}.g.{c_g}"
                save_path = os.path.join(cfg.dataset_dir, save_name)
                augmenter = Augmenter(dataset=dataset, 
                            transforms=transforms,  
                            transform_probabilities=policy_probabilities,
                            num_augmentations_per_record=cfg.augment.num_augmentations_per_record,
                            num_transforms_to_apply=cfg.augment.num_transforms_to_apply,
                            batch_size=cfg.augment.batch_size, 
                            keep_originals=cfg.augment.keep_originals)
                aug_dataset = augmenter.augment()
                aug_dataset.save_to_disk(save_path)
                torch.cuda.empty_cache()
                log.info(f"{cfg.augment.technique} augmented dataset saved @ {save_path}!")

        else:

            aggregated_performance = (cfg.fada.c_a * alignment) + \
                                     (cfg.fada.c_f * fluency) + \
                                     (cfg.fada.c_g * grammar)
            applicability_rate = np.nan_to_num(changes / counts, 0)
            tfim = softmax(applicability_rate * aggregated_performance, axis=0)

            policy_probabilities = implement_policy_probabilities(tfim, features)

            save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.{cfg.augment.technique}.a.{cfg.fada.c_a}.f.{cfg.fada.c_f}.g.{cfg.fada.c_g}"
            save_path = os.path.join(cfg.dataset_dir, save_name)
            augmenter = Augmenter(dataset=dataset, 
                        transforms=transforms,  
                        transform_probabilities=policy_probabilities,
                        num_augmentations_per_record=cfg.augment.num_augmentations_per_record,
                        num_transforms_to_apply=cfg.augment.num_transforms_to_apply,
                        batch_size=cfg.augment.batch_size, 
                        keep_originals=cfg.augment.keep_originals)
            aug_dataset = augmenter.augment()
            aug_dataset.save_to_disk(save_path)
            torch.cuda.empty_cache()
    
    log.info(f"{cfg.augment.technique} augmented dataset saved @ {save_path}!")

if __name__ == "__main__":
    fada_search()
    fada_augment()