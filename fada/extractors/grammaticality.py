import language_tool_python
import numpy as np

class GrammarMetric:
    def __init__(self) -> None:
        """
        Use language_tool_python to check grammer.
        :Package Requirements:
            * pip install language_tool_python
        :Language: english
        """
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.save_name = "grammar_score"

    def find_grammar_issues(self, text):
        return self.language_tool.check(text)

    def correct_grammar_issues(self, text):
        return self.language_tool.correct(text)
    
    def evaluate(self, dataset, annotate_dataset=False):
        scores = [len(self.find_grammar_issues(t)) for t in dataset['text']]
        if annotate_dataset:
            if self.save_name in dataset.features:
                dataset = dataset.remove_columns([self.save_name])
            dataset = dataset.add_column(self.save_name, [s for s in scores])
        return dataset, np.array(scores)
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Higher is better. Anything lower than 1 means that the
        changes made to the text reduced grammaticality. 
        """
        before_dataset, before_scores = self.evaluate(before_dataset)
        after_dataset, after_scores   = self.evaluate(after_dataset)
        scores = np.nan_to_num(before_scores.mean() / after_scores.mean())
        if annotate_after_dataset:
            if self.save_name in after_dataset.features:
                after_dataset = after_dataset.remove_columns([self.save_name])
            after_dataset = after_dataset.add_column(self.save_name, [s for s in scores])
        return after_dataset, scores


if __name__ == '__main__':
    
    from transform import Transform
    from augmenter import Augmenter
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

    g_metric = GrammarMetric()

    dataset, g_scores = g_metric.evaluate(dataset)
    print(f"original_dataset_details: {dataset}")
    print(f"original_dataset_text: {dataset['text']}")
    print(f"original_grammar_scores (raw): {g_scores}")
    print(f"original_grammar_scores (mean): {g_scores.mean()}")
    aug_dataset, g_scores = g_metric.evaluate(aug_dataset)
    print(f"augmented_dataset_details: {aug_dataset}")
    print(f"augmented_dataset_text: {aug_dataset['text']}")
    print(f"augmented_grammar_scores (raw): {g_scores}")
    print(f"augmented_grammar_scores (mean): {g_scores.mean()}")
    aug_dataset, g_scores = g_metric.evaluate_before_and_after(dataset, aug_dataset)
    print(f"augmented_dataset_details: {aug_dataset}")
    print(f"diffed_grammar_scores (raw): {g_scores}")
    print(f"diffed_grammar_scores (mean): {g_scores.mean()}")

    # (fada) C:\Users\fabri\Documents\GitHub\fada\src>python -m extractors.grammaticality
    # original_dataset_details: Dataset({
    #     features: ['text', 'label', 'idx'],
    #     num_rows: 3
    # })
    # original_dataset_text: ['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ']
    # original_grammar_scores (raw): [1 2 1]
    # original_grammar_scores (mean): 1.3333333333333333
    
    # augmented_dataset_details: Dataset({
    #     features: ['text', 'label', 'transforms_applied', 'is_changed'],
    #     num_rows: 3
    # })
    # augmented_dataset_text: ['; hide ! new secretions from the parental units ', 'muzzle no jocosity , only labored funny ', 'that bed-hop its U and scrabble something rather beautiful about human pessimism ']
    # augmented_grammar_scores (raw): [2 2 1]
    # augmented_grammar_scores (mean): 1.6666666666666667
    
    # augmented_dataset_details: Dataset({
    #     features: ['text', 'label', 'transforms_applied', 'is_changed', 'grammar_score'],
    #     num_rows: 3
    # })
    # diffed_grammar_scores (raw): [0.5 1.  1. ]
    # diffed_grammar_scores (mean): 0.8333333333333334