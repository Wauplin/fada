import torch
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter
from cleanlab.filter import find_label_issues

from utils import *


class PerformanceExtractor:
    def __init__(self, dataset_name, scorer, model_id=None):
        self.dataset_name = dataset_name
        self.scorer = scorer
        self.model_id = model_id
        self.api = HfApi()
        self.pipe = None
        self.device = 0 if torch.cuda.is_available() else -1

        if self.model_id and not self.pipe:
            self.create_pipe(self.model_id)

        if not self.pipe:
            self.find_model_for_dataset()

    def create_pipe(self, model_id):
        self.pipe = pipeline("text-classification", 
                            model=model_id, 
                            device=self.device, 
                            padding=True, 
                            truncation=True,
                            top_k=None)
        return self.pipe

    def find_model_for_dataset(self):
        model_filter = ModelFilter(
            task="text-classification",
            library="pytorch",
            # model_name=dataset_name,
            trained_dataset=self.dataset_name)
        model_id = next(iter(self.api.list_models(filter=model_filter)))
        if model_id:
            model_id = getattr(model_id, 'modelId')
            print('Using ' + model_id + ' to support evaluation.')
            self.create_pipe(model_id)

    def extract_prediction_probabilities(self, inputs):
        output = self.pipe(inputs)
        return torch.stack([vectorize(o) for o in output])
    
    def extract_prediction_classes(self, inputs):
        return torch.argmax(self.extract_prediction_probabilities(inputs), axis=1)

    def __call__(self, inputs, targets, indices=None):
        probs   = self.extract_prediction_probabilities(inputs)
        targets = torch.tensor(targets)
        return self.scorer(probs, targets, indices)


class Likelihood:
    def __init__(self):
        self.scorer = torch.nn.NLLLoss(reduction="none")
    
    def __call__(self, probs, targets, indices=None):
        return -self.scorer(probs, targets).numpy()
    
class InverseLikelihood:
    def __init__(self):
        self.scorer = torch.nn.NLLLoss(reduction="none")
    
    def __call__(self, probs, targets, indices=None):
        return 1+self.scorer(probs, targets).numpy()
    
class CleanLabSafe:
    def __init__(self):
        pass
        
    def __call__(self, probs, targets, indices=None):
        probs = probs.numpy()
        targets = targets.numpy()
        scores = ~find_label_issues(
            labels=targets,
            pred_probs=probs,
            n_jobs=1
        )
        return scores.astype(np.int32).tolist()
    
class LikelihoodShift:
    def __init__(self, original_dataset, direction="positive"):
        self.original_dataset = original_dataset
        self.direction = direction
        self.scorer = torch.nn.NLLLoss(reduction="none")
        
    def __call__(self, probs, targets, indices=None):
        new_scores  = -self.scorer(probs, targets).numpy()
        
        old_probs   = torch.tensor(self.original_dataset.select(indices)["preds"])
        old_targets = torch.tensor(self.original_dataset.select(indices)['label'])
        old_scores  = -self.scorer(old_probs, old_targets).numpy()
            
        if self.direction in "positive":
            scores = (new_scores - old_scores).clip(0, 1)
        elif self.direction in "negative":
            scores = (old_scores - new_scores).clip(0, 1)
        else:
            scores = new_scores
        return scores