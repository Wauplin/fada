import torch
import evaluate
import numpy as np
import transformers

from fada.utils import repeat_array

class FluencyMetric:
    def __init__(self, model_id='gpt2') -> None:
        """
        Use gpt2 to measure how perplexing / surprising a given text is 
        to a well trained language model. When used on text that we know
        is natural / human sounding, then perplexity is a measure of 
        model quality. However, when we trust that the language model is
        pretty good already and we aren't sure about the quality of the 
        text, then we can use perplexity to measure text naturalness. 
        :Package Requirements:
            * pip install evaluate
        :Language: english
        """
        self.model_id = model_id
        self.metric = evaluate.load("perplexity", module_type="metric")
        self.save_name = "fluency_score"
        torch.use_deterministic_algorithms(False)

    def disable_progress_bars(self):
        evaluate.utils.logging.disable_progress_bar() 
        transformers.utils.logging.disable_progress_bar() 

    def enable_progress_bars(self):
        evaluate.utils.logging.enable_progress_bar() 
        transformers.utils.logging.enable_progress_bar() 
    
    def evaluate(self, dataset, annotate_dataset=False):
        scores = self.metric.compute(
            predictions=dataset['text'], 
            model_id=self.model_id,
            max_length=256)['perplexities']
        if annotate_dataset:
            if self.save_name in dataset.features:
                dataset = dataset.remove_columns([self.save_name])
            dataset = dataset.add_column(self.save_name, [s for s in scores])
        return dataset, np.array(scores)
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Higher is better. Anything lower than 1 means that the
        changes made to the text reduced fluency. 
        """
        before_dataset, before_scores = self.evaluate(before_dataset)
        after_dataset, after_scores   = self.evaluate(after_dataset)
        before_scores = repeat_array(after_scores, before_scores)
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

    f_metric = FluencyMetric()

    dataset, f_scores = f_metric.evaluate(dataset)
    print(f"original_dataset_details: {dataset}")
    print(f"original_dataset_text: {dataset['text']}")
    print(f"original_fluency_scores (raw): {f_scores}")
    print(f"original_fluency_scores (mean): {f_scores.mean()}")
    aug_dataset, f_scores = f_metric.evaluate(aug_dataset)
    print(f"augmented_dataset_details: {aug_dataset}")
    print(f"augmented_dataset_text: {aug_dataset['text']}")
    print(f"augmented_fluency_scores (raw): {f_scores}")
    print(f"augmented_fluency_scores (mean): {f_scores.mean()}")
    aug_dataset, f_scores = f_metric.evaluate_before_and_after(dataset, aug_dataset)
    print(f"augmented_dataset_details: {aug_dataset}")
    print(f"diffed_fluency_scores (raw): {f_scores}")
    print(f"diffed_fluency_scores (mean): {f_scores.mean()}")

    # (fada) C:\Users\fabri\Documents\GitHub\fada>python -m fada.extractors.fluency
    # original_dataset_details: Dataset({
    #     features: ['text', 'label', 'idx'],
    #     num_rows: 3
    # })
    # original_dataset_text: ['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ']
    # original_fluency_scores (raw): [2923.11279297  549.8616333   364.08105469]
    # original_fluency_scores (mean): 1279.0184936523438
    
    # augmented_dataset_details: Dataset({
    #     features: ['text', 'label', 'transforms_applied', 'is_changed'],
    #     num_rows: 3
    # })
    # augmented_dataset_text: ['; hide ! new secretions from the parental units ', 'muzzle no jocosity , only labored funny ', 'that bed-hop its U and scrabble something rather beautiful about human pessimism ']
    # augmented_fluency_scores (raw): [6287.62597656 1760.31665039  858.51586914]
    # augmented_fluency_scores (mean): 2968.8194986979165
    
    # augmented_dataset_details: Dataset({
    #     features: ['text', 'label', 'transforms_applied', 'is_changed', 'fluency_score'],
    #     num_rows: 3
    # })
    # diffed_fluency_scores (raw): [0.46489928 0.31236518 0.42408192]
    # diffed_fluency_scores (mean): 0.4004487933086877