import torch
import numpy as np
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter
from cleanlab.rank import get_label_quality_scores

from fada.utils import vectorize

class AlignmentMetric:
    """
    Use cleanlab to generate a label alignment score.  
    :Package Requirements:
        * pip install cleanlab
    """
    def __init__(self, builder_name, config_name, model_id=None):
        self.builder_name = builder_name
        self.config_name = config_name
        self.model_id = model_id
        self.api = HfApi()
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = None
        self.save_name = "alignment_score"

        # initializations
        self.find_model_for_dataset()

    def find_model_for_dataset(self):
        if not self.model_id:
            if self.config_name in ["default", "plain_text"]:
                search_name = self.builder_name
            else:
                search_name = self.config_name
            
            model_filter = ModelFilter(
                task="text-classification",
                library="pytorch",
                # model_name=dataset_name,
                trained_dataset=search_name)

            model_id = next(iter(self.api.list_models(filter=model_filter)))
            model_id = getattr(model_id, 'modelId')
        else:
            model_id = self.model_id

        print('Using ' + model_id + ' to support cleanlab datalabel issues.')
        self.pipe = pipeline("text-classification", 
                            model=model_id, 
                            tokenizer=(model_id, {"max_length":512, "padding":"max_length", "truncation":True}),
                            device=self.device, 
                            top_k=None)

    def extract_prediction_probabilities(self, dataset):
        output = self.pipe(dataset['text'])
        return np.stack([vectorize(o) for o in output])

    def evaluate(self, dataset, annotate_dataset=False):
        pred_probs = self.extract_prediction_probabilities(dataset)
        scores = get_label_quality_scores(
            labels=dataset['label'],
            pred_probs=pred_probs,  
        )
        if annotate_dataset:
            if self.save_name in dataset.features:
                dataset = dataset.remove_columns([self.save_name])
            dataset = dataset.add_column(self.save_name, [s for s in scores])
        return dataset, np.array(scores)
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Higher is better. Anything lower than 1 means that the
        changes made to the text reduced label alignment. 
        """
        before_dataset, before_scores = self.evaluate(before_dataset)
        after_dataset, after_scores   = self.evaluate(after_dataset)
        scores = np.nan_to_num(after_scores / before_scores)
        if annotate_after_dataset:
            if self.save_name in after_dataset.features:
                after_dataset = after_dataset.remove_columns([self.save_name])
            after_dataset = after_dataset.add_column(self.save_name, [s for s in scores])
        return after_dataset, scores
    

if __name__ == '__main__':
    
    from fada.transform import Transform
    from fada.augmenter import Augmenter
    from datasets import load_dataset
    import sibyl
    import numpy as np

    dataset_config = ("glue", "sst2")
    task_name = "sentiment"

    dataset = load_dataset(*dataset_config, split="train[:3]")
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
        keep_originals=False,
        batch_size=batch_size)
    aug_dataset = uni_augmenter.augment()

    a_metric = AlignmentMetric()

    dataset, a_scores = a_metric.evaluate(dataset)
    print(f"original_dataset_details: {dataset}")
    print(f"original_dataset_text: {dataset['text']}")
    print(f"original_alignment_scores (raw): {a_scores}")
    print(f"original_alignment_scores (mean): {a_scores.mean()}")
    aug_dataset, a_scores = a_metric.evaluate(aug_dataset)
    print(f"augmented_dataset_details: {aug_dataset}")
    print(f"augmented_dataset_text: {aug_dataset['text']}")
    print(f"augmented_alignment_scores (raw): {a_scores}")
    print(f"augmented_alignment_scores (mean): {a_scores.mean()}")
    aug_dataset, a_scores = a_metric.evaluate_before_and_after(dataset, aug_dataset)
    print(f"augmented_dataset_details: {aug_dataset}")
    print(f"diffed_alignment_scores (raw): {a_scores}")
    print(f"diffed_alignment_scores (mean): {a_scores.mean()}")


    # (fada) C:\Users\fabri\Documents\GitHub\fada>python -m fada.extractors.alignment
    # Using distilbert-base-uncased-finetuned-sst-2-english to support cleanlab datalabel issues.
    # original_dataset_details: Dataset({
    #     features: ['text', 'label', 'idx'],
    #     num_rows: 3
    # })
    # original_dataset_text: ['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ']
    # original_alignment_scores (raw): [0.99794275 0.9997538  0.9998801 ]
    # original_alignment_scores (mean): 0.9991922378540039
    
    # augmented_dataset_details: Dataset({
    #     features: ['text', 'label', 'transforms_applied', 'is_changed'],
    #     num_rows: 3
    # })
    # augmented_dataset_text: ['; hide ! new secretions from the parental units ', 'muzzle no jocosity , only labored funny ', 'that bed-hop its U and scrabble something rather beautiful about human pessimism ']
    # augmented_alignment_scores (raw): [0.9988643  0.9612202  0.99816906]
    # augmented_alignment_scores (mean): 0.9860844612121582
    
    # augmented_dataset_details: Dataset({
    #     features: ['text', 'label', 'transforms_applied', 'is_changed', 'alignment_score'],
    #     num_rows: 3
    # })
    # diffed_alignment_scores (raw): [1.0009234  0.96145695 0.9982888 ]
    # diffed_alignment_scores (mean): 0.9868897795677185