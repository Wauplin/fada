from fada.transform import Transform
from fada.augmenter import Augmenter
from fada.utils import load_class

class CheckListAugmenter:

    def __init__(self):
        self.transforms_class_names = [
            "sibyl.transformations.text.entities.change_location.ChangeLocation",
            "sibyl.transformations.text.entities.change_name.ChangeName",
            "sibyl.transformations.text.word_swap.change_number.ChangeNumber",
            "sibyl.transformations.text.contraction.contract_contractions.ContractContractions",
            "sibyl.transformations.text.contraction.expand_contractions.ExpandContractions"
        ]
        self.transforms = [load_class(t) for t in self.transforms_class_names]
        self.transforms = sorted(self.transforms, key=lambda t: t.__name__)
        self.transforms = [Transform(t) for t in self.transforms]

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

if __name__ == "__main__":
    
    from datasets import load_dataset, load_from_disk

    print("Testing CheckListAugmenter...")
    augmenter = CheckListAugmenter()

    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.rename_column("sentence", "text").select(range(1000))
    print(dataset)

    aug_dataset = augmenter(dataset, num_aug=3)
    print(aug_dataset)