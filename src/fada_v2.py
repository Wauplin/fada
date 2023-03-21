# data
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from datasets import load_dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar

# amrs
import amrlib
import penman

# transform
import sibyl
import torch
import inspect
import random
from functools import partial

# eval pipeline
import pandas as pd
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.special import softmax

# train pipeline
import shutil
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)

# cleanlab pipeline
from cleanlab.filter import find_label_issues

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from transform import *
from featurizers.amr import *
from utils import *
from augmenter import *

if __name__ == "__main__":

    class Likelihood:
        def __init__(self):
            self.scorer = torch.nn.NLLLoss(reduction="none")
        
        def __call__(self, probs, targets):
            scores = -self.scorer(probs, targets).numpy()
            return scores.tolist()
        
    class InverseLikelihood:
        def __init__(self):
            self.scorer = torch.nn.NLLLoss(reduction="none")
        
        def __call__(self, probs, targets):
            scores = 1+self.scorer(probs, targets).numpy()
            return scores.tolist()
        
    class CleanLabSafe:
        def __init__(self):
            pass
            
        def __call__(self, probs, targets):
            print(probs, targets)
            probs = probs.numpy()
            targets = targets.numpy()
            scores = ~find_label_issues(
                labels=targets,
                pred_probs=probs,
                n_jobs=1
            )
            return scores.astype(np.int32).tolist()

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

        def __call__(self, inputs, targets):
            probs   = self.extract_prediction_probabilities(inputs)
            targets = torch.tensor(targets)
            return self.scorer(probs, targets)


    # code to run

    torch.use_deterministic_algorithms(False)

    dataset_config = ("glue", "sst2")
    task_name = "sentiment"

    dataset = load_dataset(*dataset_config, split="train")
    dataset = dataset.rename_column("sentence", "text")

    blacklist = [
        sibyl.Emojify,
        sibyl.AddPositiveEmoji,
        sibyl.AddNegativeEmoji,
        sibyl.Demojify,
        sibyl.RemovePositiveEmoji,
        sibyl.RemoveNegativeEmoji,
        sibyl.AddPositiveEmoji,
        sibyl.AddNegativeEmoji,
        sibyl.InsertPositivePhrase,
        sibyl.InsertNegativePhrase,
        sibyl.AddPositiveLink,
        sibyl.AddNegativeLink,
        sibyl.ImportLinkText,
        sibyl.AddNegation,
        sibyl.RemoveNegation,
        sibyl.ChangeAntonym,
        sibyl.ConceptMix,
        sibyl.TextMix,
        sibyl.SentMix,
        sibyl.WordMix,
        sibyl.Concept2Sentence
    ]
    transforms = [t for t in sibyl.TRANSFORMATIONS if t not in blacklist]
    transforms = sorted(transforms, key=lambda t: t.__name__)
    transforms = [Transform(t, task_name=task_name) for t in transforms]

    feature_extractor = AMRFeatureExtractor()
    perf_extractor = PerformanceExtractor(dataset.builder_name, CleanLabSafe())

    dataset = load_from_disk("./datasets/glue.sst2.featurized")
    features = np.array(dataset["features"])

    save_dir       = "./fadata/v2/"

    num_rows       = len(dataset)
    num_transforms = len(transforms)
    num_features   = len(feature_extractor.featurizers)

    performances   = np.zeros((num_transforms, num_features))
    counts         = np.zeros((num_transforms, num_features))
    changes        = np.zeros((num_transforms, num_features))
    policy         = np.full((num_transforms, num_features), fill_value=1/num_transforms)

    min_coverage = 100
    num_to_transform_per_step = 25

    policy_difference = np.inf
    convergence_threshold = 1 / (num_transforms + num_features)

    i = 0
    while policy_difference > convergence_threshold:

        # find low coverage (t,f) pairs
        ts, fs   = np.where(changes < min_coverage)
        tf_pairs = list(zip(ts, fs))

        for t, f in tqdm(tf_pairs):

            f_candidates  = np.where(features[:,f] == 1)[0]

            # feature missing in dataset
            if not f_candidates.size:
                continue

            num_to_sample = num_to_transform_per_step if len(f_candidates) > num_to_transform_per_step else len(f_candidates)
            f_indices     = np.random.choice(f_candidates, num_to_sample, replace=False)
            f_dataset     = dataset.select(f_indices)

            t_prob = np.zeros(num_transforms)
            t_prob[t] = 1
            transform_probabilities = np.array([t_prob for _ in range(f_dataset.num_rows)])

            print(f_dataset)
            print(f_dataset[0])

            augmenter = Augmenter(dataset=f_dataset, 
                          transforms=transforms,  
                          transform_probabilities=transform_probabilities,
                          num_augmentations_per_record=1,
                          num_transforms_to_apply=1,
                          batch_size=10, 
                          keep_originals=False)
            aug_dataset = augmenter.augment()

            try:
                performance = perf_extractor(f_dataset["text"], f_dataset["label"])
            except Exception as e:
                print(e)
                performance = [1] * f_dataset.num_rows
            aug_dataset = aug_dataset.add_column("performance", [p for p in performance])

            performance = np.array(aug_dataset["performance"]).sum()
            num_changes = np.array(aug_dataset["is_changed"]).sum()

            counts[t,f]       += f_dataset.num_rows
            changes[t,f]      += num_changes  
            performances[t,f] += performance   

        # compute augmentation policy
        average_performance    = np.nan_to_num(performances / counts, 0)
        applicability_rate     = np.nan_to_num(changes / counts, 0)
        new_policy             = softmax(average_performance * applicability_rate, axis=0)

        policy_difference      = np.linalg.norm(new_policy - policy)
        policy                 = new_policy

        print(f"policy_difference: {policy_difference} (convergence_threshold: {convergence_threshold})")

        policy_heatmap(policy, transforms, feature_extractor.featurizers)

        print("Saving intermediate matrices...")
        np.save(os.path.join(save_dir, f"glue.sst2.fada.v2.counts-step-{i}"), counts)
        np.save(os.path.join(save_dir, f"glue.sst2.fada.v2.changes-step-{i}"), changes)
        np.save(os.path.join(save_dir, f"glue.sst2.fada.v2.performances-step-{i}"), performances)
        np.save(os.path.join(save_dir, f"glue.sst2.fada.v2.policy-step-{i}"), policy)

        i += 1