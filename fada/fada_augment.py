import os
import glob
import logging
import torch
import numpy as np
from scipy.special import softmax
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import load_from_disk

from fada.transform import Transform
from fada.augmenter import Augmenter
from fada.utils import implement_policy_probabilities, load_class
from fada.filters import balance_dataset

from fada.augmenters import (
    EDANLPAugmentor,
    CheckListAugmenter,
    TextAutoAugmenter,
    UniformAugmenter, 
    FADAAugmenter
)

log = logging.getLogger(__name__)
torch.use_deterministic_algorithms(False)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def fada_augment(cfg: DictConfig) -> None:

    log.info("Starting fada_augment.")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Setting up working directories.")
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.augment.save_dir, exist_ok=True)

    dataset_matcher = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.*.{cfg.dataset.num_per_class}*{cfg.augment.tfim_dataset_builder_name if 'transfer' in cfg.augment.technique else ''}"
    log.info(f"Checking to see if fada_augment has been run for this dataset ({dataset_matcher}) before...")
    dataset_paths = glob.glob(os.path.join(cfg.augment.save_dir, dataset_matcher))
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
    num_transforms = len(transforms)
    log.info(f"Running augment with {num_transforms} to choose from...")
    
    log.info(f"Constructing original (unaugmented) dataset...")
    save_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.original.{cfg.dataset.num_per_class}".replace("/", ".")
    save_path = os.path.join(cfg.augment.save_dir, save_name)

    if not os.path.exists(save_path):
        annotated_dataset_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.annotated".replace("/", ".")
        annotated_dataset_path = os.path.join(cfg.dataset_dir, annotated_dataset_name)
        if os.path.exists(annotated_dataset_path):
            dataset = load_from_disk(annotated_dataset_path)
            features = np.array(dataset["features"])

        dataset_needs_balancing = False
        if isinstance(cfg.dataset.num_per_class, int):
            if cfg.dataset.num_per_class > 0:
                num_per_class = cfg.dataset.num_per_class
                dataset_needs_balancing = True
        elif str(cfg.dataset.num_per_class) == "infer" and dataset.num_rows > cfg.dataset.max_size:
            num_labels = len(np.unique(dataset['label']))
            num_per_class = int(cfg.dataset.max_size / num_labels)
            dataset_needs_balancing = True

        if dataset_needs_balancing:
            log.info(f"Balancing dataset with num_per_class={num_per_class}")
            dataset = balance_dataset(dataset, num_per_class=num_per_class)

        dataset.save_to_disk(save_path)
        log.info(f"Original dataset saved @ {save_path}!")
    else:
        log.info(f"found {save_path}... loading from disk...")
        dataset = load_from_disk(save_path)
    log.info(dataset)

    log.info("Removing features from original dataset.")
    keep_cols = ['text', 'label', 'idx']
    features = dataset["features"]
    dataset = dataset.remove_columns([c for c in dataset.features.keys() if c not in keep_cols])
    log.info(dataset)

    log.info(f"Initializing Augmenters...")

    eda_augmenter       = EDANLPAugmentor()
    checklist_augmenter = CheckListAugmenter()
    taa_augmenter       = TextAutoAugmenter()
    uniform20_augmenter = UniformAugmenter(transforms=transforms)
    fada20_augmenter    = FADAAugmenter(transforms=transforms, cfg=cfg)

    log.info(f"Beginning EDA augmentation...")
    eda_dataset_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.eda{num_transforms}.{cfg.dataset.num_per_class}".replace("/", ".")
    eda_save_path = os.path.join(cfg.augment.save_dir, eda_dataset_name)
    
    if not os.path.exists(eda_save_path):
        eda_dataset = eda_augmenter(
            dataset=dataset,  
            num_aug=cfg.augment.num_augmentations_per_record
        )
        eda_dataset.save_to_disk(eda_save_path)
        torch.cuda.empty_cache()
        log.info(f"EDA augmented dataset saved @ {eda_save_path}!")
        log.info(eda_dataset)
    else:
        log.info(f"found {eda_save_path}... skipping...")

    log.info(f"Beginning CheckList augmentation...")
    checklist_dataset_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.checklist{num_transforms}.{cfg.dataset.num_per_class}".replace("/", ".")
    checklist_save_path = os.path.join(cfg.augment.save_dir, checklist_dataset_name)
    if not os.path.exists(checklist_save_path):
        checklist_dataset = checklist_augmenter(
            dataset=dataset,  
            num_aug=cfg.augment.num_augmentations_per_record
        )
        checklist_dataset.save_to_disk(checklist_save_path)
        torch.cuda.empty_cache()
        log.info(f"CheckList augmented dataset saved @ {checklist_save_path}!")
        log.info(checklist_dataset)
    else:
        log.info(f"found {checklist_save_path}... skipping...")

    log.info(f"Beginning TAA augmentation...")
    taa_dataset_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.taa{num_transforms}.{cfg.dataset.num_per_class}".replace("/", ".")
    taa_save_path = os.path.join(cfg.augment.save_dir, taa_dataset_name)
    if not os.path.exists(taa_save_path):
        if "glue" in cfg.dataset.builder_name:
            name = cfg.dataset.config_name
        elif "sst5" in cfg.dataset.builder_name:
            name = "sst5"
        else:
            name = cfg.dataset.builder_name
        taa_dataset = taa_augmenter(
            dataset=dataset,  
            name=name,
            num_aug=cfg.augment.num_augmentations_per_record
        )
        taa_dataset.save_to_disk(taa_save_path)
        torch.cuda.empty_cache()
        log.info(f"TAA augmented dataset saved @ {taa_save_path}!")
        log.info(taa_dataset)
    else:
        log.info(f"found {taa_save_path}... skipping...")

    log.info(f"Beginning uniform augmentation...")
    uniform_dataset_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.uniform20{num_transforms}.{cfg.dataset.num_per_class}".replace("/", ".")
    uniform_save_path = os.path.join(cfg.augment.save_dir, uniform_dataset_name)
    if not os.path.exists(uniform_save_path):
        uniform_dataset = uniform20_augmenter(
            dataset=dataset,  
            num_aug=cfg.augment.num_augmentations_per_record, 
            num_transforms_to_apply=cfg.augment.num_transforms_to_apply, 
            batch_size=cfg.augment.batch_size, 
            keep_originals=cfg.augment.keep_originals
        )
        uniform_dataset.save_to_disk(uniform_save_path)
        torch.cuda.empty_cache()
        log.info(f"Uniform augmented dataset saved @ {uniform_save_path}!")
        log.info(uniform_dataset)
    else:
        log.info(f"found {uniform_save_path}... skipping...")

    log.info(f"Beginning FADA augmentation...")
    fada_dataset_name = f"{cfg.dataset.builder_name}.{cfg.dataset.config_name}.fada20{num_transforms}.{cfg.dataset.num_per_class}".replace("/", ".")
    fada_save_path = os.path.join(cfg.augment.save_dir, fada_dataset_name)
    if not os.path.exists(fada_save_path):
        fada_dataset = fada20_augmenter(
            dataset=dataset,  
            features=features,
            num_aug=cfg.augment.num_augmentations_per_record, 
            num_transforms_to_apply=cfg.augment.num_transforms_to_apply, 
            batch_size=cfg.augment.batch_size, 
            keep_originals=cfg.augment.keep_originals
        )
        fada_dataset.save_to_disk(fada_save_path)
        torch.cuda.empty_cache()
        log.info(f"FADA augmented dataset saved @ {fada_save_path}!")
        log.info(fada_dataset)
    else:
        log.info(f"found {fada_save_path}... skipping...")

if __name__ == "__main__":
    fada_augment()