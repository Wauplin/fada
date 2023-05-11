from fada.utils import *

class Augmenter:
    """
    Helps control the data transformation / augmentation process. 

    Parameters
    ----------
    dataset : datasets.Dataset
        The huggingface dataset with N rows to be transformed. 
    transforms : list[transform.Transform]
        The `transform.Transform` wrapped classes that alter the text 
        and potentially the label. Must be initialized already. 
    transform_probabilities : np.array, optional
        The sampling distribution over T transforms. 
            * if None, then we assume a uniform sampling. 
            * if a single probability T-length vector is provided, 
              then we apply it to all N rows in the dataset.
            * if an (N, T) matrix is provided, this indicates that each
              data point has its own sampling distribution.
    num_augmentations_per_record : int
        The number of new datapoints created for each original row in
        the provided dataset.
    num_transforms_to_apply : int
        The number of transforms to apply to each row in the dataset
    batch_size : int
        The number of rows to process per iteration. Especially important 
        to tune this for transforms that operate on batches, such as 
        `sibyl.TextMix`.
    allow_resampling : bool
        If `num_transforms_to_apply` > 1, this allows for the same 
        transform to be sampled multiple times and applied to the same
        data point. Only a handful of transforms are likely to benefit 
        from `allow_resampling=True`. Most others will likely have no 
        effect (wasteful computation), or will degrade text quality 
        (e.g. grammaticality, fluency)
    feature_extractor : class
        Adds feature information to each record in the processed 
        dataset (e.g. AMRFeatureExtractor).
    perf_extractor : class
        Adds performance infromation to each record in the processed
        dataset (e.g. PerformanceExtractor(Likelihood())).
    """
    def __init__(self, 
                 dataset,
                 transforms,
                 transform_probabilities = None, 
                 num_augmentations_per_record = 5,
                 num_transforms_to_apply = 2,
                 batch_size = 10,
                 allow_resampling = False,
                 keep_originals = True,
                 feature_extractor = None,
                 perf_extractor = None):
        
        self.dataset = dataset
        self.transforms = transforms
        self.transform_probabilities = transform_probabilities
        self.num_augmentations_per_record = num_augmentations_per_record
        self.num_transforms_to_apply = num_transforms_to_apply
        self.batch_size = batch_size
        self.allow_resampling = allow_resampling
        self.keep_originals = keep_originals
        self.feature_extractor = feature_extractor
        self.perf_extractor = perf_extractor
        
        # initializations
        if "idx" in self.dataset.column_names:
            self.dataset = dataset.remove_columns("idx")
        self.add_idx_to_dataset()
        self.num_transforms_available = len(self.transforms)
        
        if self.transform_probabilities is None:
            # set to uniform
            num_examples = len(self.dataset)
            num_transforms = len(self.transforms)
            uniform_policy = np.full((num_examples, num_transforms), fill_value=1/num_transforms)
            self.transform_probabilities = uniform_policy

        elif len(self.transform_probabilities.shape) == 1:
            num_examples = len(self.dataset)
            num_transforms = len(self.transforms)
            given_policy = np.full((num_examples, num_transforms), fill_value=self.transform_probabilities)
            self.transform_probabilities = given_policy
        
    def add_idx_to_dataset(self):
        if 'idx' not in self.dataset.column_names:
            self.dataset = self.dataset.add_column("idx", range(len(self.dataset)))
        
    def apply_to_batch(self, batch):
        new_texts, new_labels, transforms_applied, is_changed = [], [], [], []
        for idx, text, label in zip(batch['idx'], batch['text'], batch['label']):
            actual_batch_size = len(batch['idx'])
            original_text, original_label = text, label
            for _ in range(self.num_augmentations_per_record):
                sampled_transforms = sample_transforms(self.transforms, 
                                                       p=self.transform_probabilities[idx], 
                                                       n=self.num_transforms_to_apply, 
                                                       replace=self.allow_resampling)
                transforms_applied.append(transforms_to_ids(sampled_transforms, self.transforms))
                for t in sampled_transforms:
                    try:
                        text, label = t.apply([text], [label])
                        text, label = text[0], label[0]
                    except Exception as e: 
                        print(e)
                        print(f"[Augmenter]: skipping augmentation from {t.transform_class.__name__} on text:'{text}' and label: {label}")

                # avoid adding records with empty text
                if text:
                    new_texts.append(text)
                    new_labels.append(label)
                    is_changed.append(int(original_text != text))

        if self.keep_originals:
            new_texts = batch['text'] + new_texts
            new_labels = batch['label'] + new_labels
            realized_batch_size = len(new_labels)
            transforms_applied = transforms_applied + np.zeros((actual_batch_size, len(self.transforms)), dtype=np.int64).tolist()
            is_changed = is_changed + [0] * actual_batch_size
            out = {
                "text": new_texts, 
                "label": new_labels,
                "idx": list(range(realized_batch_size)),
                "transforms_applied": [t for t in transforms_applied],
                "is_changed": is_changed
            }
        else:
            out = {
                "text": new_texts, 
                "label": new_labels, 
                "idx": list(range(len(new_labels))),
                "transforms_applied": transforms_applied,
                "is_changed": is_changed
            }

        return out
            
                                                   
    def augment(self):
        dataset = self.dataset.map(self.apply_to_batch, batched=True, batch_size=self.batch_size)
        dataset = dataset.remove_columns("idx")
    
        # feature extraction
        if self.feature_extractor is not None:
            features = self.feature_extractor(dataset["text"])
            dataset = dataset.add_column("features", [f for f in features])
                
        # performance scoring    
        if self.perf_extractor is not None:
            performances = self.perf_extractor(dataset["text"], dataset["label"])
            dataset = dataset.add_column("performance", [p for p in performances])
                
        return dataset

if __name__ == '__main__':

    from fada.transform import Transform
    from datasets import load_dataset
    import sibyl
    import numpy as np

    dataset_config = ("glue", "sst2")
    task_name = "sentiment"

    dataset = load_dataset(*dataset_config, split="train[:1000]")
    dataset = dataset.rename_column("sentence", "text")

    transforms = [
        sibyl.ChangeHypernym,
        sibyl.ChangeHyponym,
        sibyl.InsertPunctuationMarks
    ]
    transforms = [Transform(t, task_name=task_name) for t in transforms]

    num_augmentations_per_record = 1
    num_transforms_to_apply = 1
    batch_size = 1

    # uniform sampling probabilities
    uni_augmenter = Augmenter(
        dataset=dataset, 
        transforms=transforms,  
        transform_probabilities=None,
        num_augmentations_per_record=num_augmentations_per_record,
        num_transforms_to_apply=num_transforms_to_apply,
        batch_size=batch_size)
    uni_dataset = uni_augmenter.augment()

    print(uni_dataset)
    print(compare_policy_probs(uni_augmenter.transform_probabilities, uni_dataset))

    # given transform sampling probabilities
    giv_augmenter = Augmenter(
        dataset=dataset, 
        transforms=transforms,  
        transform_probabilities=np.array([0.1, 0.1, 0.8]),
        num_augmentations_per_record=num_augmentations_per_record,
        num_transforms_to_apply=num_transforms_to_apply,
        batch_size=batch_size)
    giv_dataset = giv_augmenter.augment()

    print(giv_dataset)
    print(compare_policy_probs(giv_augmenter.transform_probabilities, giv_dataset))