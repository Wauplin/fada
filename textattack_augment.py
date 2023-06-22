class AugMapper:
    def __init__(self, augmenter):
        self.augmenter = augmenter # textattack augmenter recipe

    def apply_to_batch(self, batch):
        new_texts, new_labels = [], []
        for text, label in zip(batch['text'], batch['label']):
            augmented_text = self.augmenter.augment(text)
            new_texts.extend(augmented_text)
            new_labels.extend([label] * len(augmented_text))

        return {
            "text": new_texts,
            "label": new_labels,
            "idx": list(range(len(new_labels))),
        }

if __name__ == "__main__":
    import os
    import glob
    from datasets import load_dataset, load_from_disk
    from textattack.augmentation.recipes import (
        EasyDataAugmenter,
        CheckListAugmenter,
    )

    # checklist
    checklist_augmenter = CheckListAugmenter(
        transformations_per_example=3
    )
    checklist_aug_mapper = AugMapper(checklist_augmenter)

    # eda
    eda_augmenter = EasyDataAugmenter(
        pct_words_to_swap=0.1,
        transformations_per_example=3
    )
    eda_aug_mapper = AugMapper(eda_augmenter)

    dataset_paths = glob.glob("./fada/fadata/datasets/*original*")
    dataset_paths.sort()

    for dataset_path in dataset_paths:

        print(dataset_path)

        dataset = load_from_disk(dataset_path)

        # augment + save checklist
        checklist_save_path = dataset_path.replace("original", "checklist")
        if os.path.exists(checklist_save_path):
            print(f"found existing dataset {checklist_save_path}... skipping...") 
        else:
            checklist_dataset = dataset.map(checklist_aug_mapper.apply_to_batch, batched=True, batch_size=10)
            checklist_dataset.save_to_disk(checklist_save_path)
            
        # augment + save eda
        eda_save_path = dataset_path.replace("original", "eda")
        if os.path.exists(eda_save_path):
            print(f"found existing dataset {eda_save_path}... skipping...") 
        else:
            eda_dataset = dataset.map(eda_aug_mapper.apply_to_batch, batched=True, batch_size=10)
            eda_dataset.save_to_disk(eda_save_path)
