from fada.transform import Transform
from fada.augmenter import Augmenter
from fada.utils import implement_policy_probabilities
from scipy.special import softmax
import numpy as np
import os
import glob

class UniformAugmenter:

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, dataset, num_aug=3, num_transforms_to_apply=1, batch_size=8, keep_originals=True):

        augmenter = Augmenter(dataset=dataset, 
                    transforms=self.transforms,  
                    transform_probabilities=None,
                    num_augmentations_per_record=num_aug,
                    num_transforms_to_apply=num_transforms_to_apply,
                    batch_size=batch_size, 
                    keep_originals=keep_originals)
        aug_dataset = augmenter.augment()
        return aug_dataset

class FADAAugmenter:

    def __init__(self, transforms=None, tfim=None, cfg=None):
        self.transforms = transforms
        self.tfim = tfim
        self.cfg = cfg

    def reweight_tfim(self, weights : dict):

        tfim_matcher = os.path.join(cfg.fada.tfim_dir, f"{cfg.augment.tfim_dataset_builder_name}.{cfg.augment.tfim_dataset_config_name}*")
        matrix_paths = glob.glob(tfim_matcher)
        max_id = int(max([m.split("-")[-1].split(".")[0] for m in matrix_paths]))

        save_name = f"{cfg.augment.tfim_dataset_builder_name}.{cfg.augment.tfim_dataset_config_name}.fada20"

        counts    = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.counts-step-{max_id}.npy"))
        changes   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.changes-step-{max_id}.npy"))
        alignment = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.alignment-step-{max_id}.npy"))
        fluency   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.fluency-step-{max_id}.npy"))
        grammar   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.grammar-step-{max_id}.npy"))
        div_sem   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.div_sem-step-{max_id}.npy"))
        div_syn   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.div_syn-step-{max_id}.npy"))
        div_mor   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.div_mor-step-{max_id}.npy"))
        div_mtr   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.div_mtr-step-{max_id}.npy"))
        div_ubi   = np.load(os.path.join(self.cfg.fada.tfim_dir, f"{save_name}.div_ubi-step-{max_id}.npy"))

        aggregated_performance = (weights['alignment'] * alignment) + \
                                 (weights['fluency'] * fluency) + \
                                 (weights['grammar'] * grammar) + \
                                 (weights['div_sem'] * div_sem) + \
                                 (weights['div_syn'] * div_syn) + \
                                 (weights['div_mor'] * div_mor) + \
                                 (weights['div_mtr'] * div_mtr) + \
                                 (weights['div_ubi'] * div_ubi) 

        applicability_rate = np.nan_to_num(changes / counts, 0)
        self.tfim = softmax(applicability_rate * aggregated_performance, axis=0)
        return self.tfim


    def __call__(self, dataset, num_aug=3, num_transforms_to_apply=1, batch_size=8, keep_originals=True):

        features = np.array(dataset["features"])
        policy_probabilities = implement_policy_probabilities(self.tfim, features)

        augmenter = Augmenter(dataset=dataset, 
                    transforms=self.transforms,  
                    transform_probabilities=policy_probabilities,
                    num_augmentations_per_record=num_aug,
                    num_transforms_to_apply=num_transforms_to_apply,
                    batch_size=batch_size, 
                    keep_originals=keep_originals)
        aug_dataset = augmenter.augment()
        return aug_dataset

if __name__ == "__main__":
    
    from datasets import load_dataset, load_from_disk
    from fada.utils import load_class
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt 

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            'dataset.builder_name=glue',
            'dataset.config_name=sst2'])

    transforms = [load_class(t) for t in cfg.transforms]
    transforms = sorted(transforms, key=lambda t: t.__name__)
    transforms = [Transform(t, task_name=cfg.dataset.task_name) for t in transforms]

    print("Testing UniformAugmenter...")
    augmenter = UniformAugmenter(transforms=transforms)

    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.rename_column("sentence", "text").select(range(1000))
    print(dataset)

    aug_dataset = augmenter(dataset, num_aug=3)
    print(aug_dataset)

    
    print("Testing FADAAugmenter...")
    import seaborn as sns

    dataset = load_from_disk('./fada/fadata/datasets/glue.sst2.annotated').select(range(1000))
    print(dataset)

    augmenter = FADAAugmenter(transforms=transforms, cfg=cfg)

    new_weights = {
        'alignment': 0.3,
        'fluency': 0.1,
        'grammar': 0.2,
        'div_sem': 0.2,
        'div_syn': 0.05,
        'div_mor': 0.05,
        'div_mtr': 0.05,
        'div_ubi': 0.05,
    }
    augmenter.reweight_tfim(new_weights)
    sns.heatmap(augmenter.tfim)
    plt.show()
    aug_dataset = augmenter(dataset, num_aug=3)
    print(aug_dataset)

    
    new_weights = {
        'alignment': 0.16667,
        'fluency': 0.16667,
        'grammar': 0.16667,
        'div_sem': 0.1,
        'div_syn': 0.1,
        'div_mor': 0.1,
        'div_mtr': 0.1,
        'div_ubi': 0.1,
    }
    augmenter.reweight_tfim(new_weights)
    sns.heatmap(augmenter.tfim)
    plt.show()
    aug_dataset = augmenter(dataset, num_aug=3)
    print(aug_dataset)

