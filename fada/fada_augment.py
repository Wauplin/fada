import os
import importlib
import glob
import logging
import torch
import numpy as np
from scipy.special import softmax
import hydra
from omegaconf import DictConfig

from datasets import load_from_disk

from fada.transform import Transform
from fada.augmenter import Augmenter
from fada.utils import implement_policy_probabilities
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
def fada_augment(cfg: DictConfig) -> None:

    log.info("Starting fada_augment.")

    log.info("Setting up working directories.")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.fada.tfim_dir, exist_ok=True)

    dataset_matcher = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.{cfg.augment.technique}.{cfg.dataset.num_per_class}*"
    log.info(f"Checking to see if fada_augment has been run for this dataset ({dataset_matcher}) before...")
    dataset_paths = glob.glob(os.path.join(cfg.dataset_dir, dataset_matcher))
    if len(dataset_paths) > 0:
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
    annotated_dataset_path = os.path.join(cfg.dataset_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated")
    if os.path.exists(annotated_dataset_path):
        dataset = load_from_disk(annotated_dataset_path)
        features = np.array(dataset["features"])
        keep_cols = ['text', 'label', 'idx']
        dataset = dataset.remove_columns([c for c in dataset.features.keys() if c not in keep_cols])

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

    log.info(f"Constructing original (unaugmented) dataset...")
    save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.original.{cfg.dataset.num_per_class}"
    save_path = os.path.join(cfg.dataset_dir, save_name)
    dataset.save_to_disk(save_path)
    log.info(f"Original dataset saved @ {save_path}!")

    log.info(f"Beginning augmentation for technique={cfg.augment.technique}")

    if "uniform" in cfg.augment.technique:
        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.{cfg.augment.technique}.{cfg.dataset.num_per_class}"
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
        
    if "fada" in cfg.augment.technique or "all" in cfg.augment.technique:

        log.info("Loading final TFIM component matrices.")
        matrix_paths = glob.glob(os.path.join(cfg.fada.tfim_dir, f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}*"))
        max_id = int(max([m.split("-")[-1].split(".")[0] for m in matrix_paths]))

        save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada"
        counts    = np.load(os.path.join(cfg.fada.tfim_dir, f"{save_name}.counts-step-{max_id}.npy"))
        changes   = np.load(os.path.join(cfg.fada.tfim_dir, f"{save_name}.changes-step-{max_id}.npy"))
        alignment = np.load(os.path.join(cfg.fada.tfim_dir, f"{save_name}.alignment-step-{max_id}.npy"))
        fluency   = np.load(os.path.join(cfg.fada.tfim_dir, f"{save_name}.fluency-step-{max_id}.npy"))
        grammar   = np.load(os.path.join(cfg.fada.tfim_dir, f"{save_name}.grammar-step-{max_id}.npy"))
        tfim      = np.load(os.path.join(cfg.fada.tfim_dir, f"{save_name}.tfim-step-{max_id}.npy"))

        if cfg.augment.technique == "fada":

            aggregated_performance = (cfg.fada.c_a * alignment) + \
                                     (cfg.fada.c_f * fluency) + \
                                     (cfg.fada.c_g * grammar)
            applicability_rate = np.nan_to_num(changes / counts, 0)
            tfim = softmax(applicability_rate * aggregated_performance, axis=0)

            policy_probabilities = implement_policy_probabilities(tfim, features)

            save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada.{cfg.dataset.num_per_class}.a.{cfg.fada.c_a}.f.{cfg.fada.c_f}.g.{cfg.fada.c_g}"
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
            
        if cfg.augment.technique == "fada-sweep":

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

                save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada.{cfg.dataset.num_per_class}.a.{c_a}.f.{c_f}.g.{c_g}"
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


        if "all" in cfg.augment.technique:

            log.info("Commencing with creating all augmented datasets - uniform + fada-sweep")

            log.info("Starting uniform aug...")
            save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.uniform.{cfg.dataset.num_per_class}"
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
            log.info(f"uniform augmented dataset saved @ {save_path}!")
            log.info(aug_dataset[0])
            
            log.info("Starting fada augs...")

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

                save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada.{cfg.dataset.num_per_class}.a.{c_a}.f.{c_f}.g.{c_g}"
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
                log.info(f"fada augmented dataset saved @ {save_path}!")
                log.info(aug_dataset[0])
    
    log.info(f"{cfg.augment.technique} augmented dataset completed!")

if __name__ == "__main__":
    fada_augment()