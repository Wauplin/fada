import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
from collections import Counter
import os
import shutil

# helper functions

def load_class(module_class_str):
    parts = module_class_str.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls

def normalize_minmax(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())  
    return df

def normalize_sum(df):
    for column in df.columns:
        df[column] = df[column] / df[column].sum()
    return df

def augment_data(batch, transform, keep_originals=True):
    new_texts, new_labels = [], []
    for text, label in zip(batch['text'], batch['label']):
        new_text, new_label = transform.apply([text], [label])
        new_texts.extend(new_text)
        new_labels.extend(new_label)
    if keep_originals:
        return {"text": batch['text'] + new_texts, "label": batch['label'] + new_labels}
    else:
        return {"text": new_texts, "label": new_labels}
    
def percent_dataset_changed(d1, d2):
    return sum([t1['text'] != t2['text'] for t1, t2 in zip(d1, d2)]) / len(d1) 

def vectorize(output):
    sorted_output = sorted(output, key=lambda d: d['label']) 
    probs = torch.tensor([d['score'] for d in sorted_output])
    return probs

def sample_transforms(transforms, p, n=2, replace=False):
    return np.random.choice(transforms, size=n, p=p, replace=replace).tolist()

def policy_heatmap(policy, transforms, featurizers):
    t_names = [t.transform_class.__name__ for t in transforms]
    f_names = [f.__name__ for f in featurizers]
    df = pd.DataFrame(policy)
    df.columns = f_names
    df.index = t_names
    sns.heatmap(df)
    plt.show(block=False)
    
def implement_policy_probabilities(policy, features):
    default_probability = policy.mean(axis=1)
    policy_probs = []
    for f in features:
        available_features = np.nonzero(f)[0]
        if len(available_features) == 0:
            probs = default_probability
        else:
            probs = policy[:, available_features].mean(axis=1)
        policy_probs.append(probs)
    return np.array(policy_probs)

def transforms_to_ids(sampled_transforms, all_transforms):
    transforms_ids = [all_transforms.index(i) for i in sampled_transforms]
    transforms_applied = np.zeros(len(all_transforms), dtype=np.int64)
    transforms_applied[transforms_ids] = 1
    return transforms_applied

def prepare_splits(dataset_dict, train_val_split = 0.9, val_test_split = 0.5, seed=10):
    has_train = has_val = has_test = False
    train_id, val_id, test_id = "train", "valid", "test"
    for split_name in list(dataset_dict.keys()):
        if "train" in split_name:
            has_train = True
            train_id = split_name
        elif "val" in split_name:
            has_val = True
            val_id = split_name
        elif "test" in split_name:
            has_test = True
            test_id = split_name
        else:
            dataset_dict.pop(split_name)

    if has_train and has_val and has_test:
        return dataset_dict
    if has_val and not has_test:
        val_test      = dataset_dict[val_id].train_test_split(train_size=val_test_split, seed=seed)
        train_dataset = dataset_dict[train_id]
        val_dataset   = val_test['train']
        test_dataset  = val_test['test']
    if has_test and not has_val:
        train_val     = dataset_dict[train_id].train_test_split(train_size=train_val_split, seed=seed)
        train_dataset = train_val['train']
        val_dataset   = train_val['test']
        test_dataset  = dataset_dict[test_id]
    if not has_val and not has_test:
        train_val     = dataset_dict[train_id].train_test_split(train_size=train_val_split, seed=seed)
        val_test      = train_val['test'].train_test_split(train_size=val_test_split, seed=seed)
        train_dataset = train_val['train']
        val_dataset   = val_test['train']
        test_dataset  = val_test['test']

    dataset_dict["train"]      = train_dataset
    dataset_dict["validation"] = val_dataset
    dataset_dict["test"]       = test_dataset

    return dataset_dict

def rename_text_columns(dataset_dict, remove_unused=True):
    text_columns = ["sentence"]
    val_columns = ["val", "valid"]
    label_columns = ["coarse_label"]
    for split_name, dataset in dataset_dict.items():
        if dataset.builder_name == "yahoo_answers_topics":
            dataset_dict[split_name] = dataset.map(
                lambda example : {'text' : example['question_title'] + " " + 
                                           example['question_content'] + " " +
                                           example['best_answer'],
                                 'label': example['topic']})  
        for column in dataset.features:
            if column in text_columns:
                dataset_dict[split_name] = dataset.rename_column(column, "text")
            if column in val_columns:
                dataset_dict[split_name] = dataset.rename_column(column, "validation")
            if column in label_columns:
                dataset_dict[split_name] = dataset.rename_column(column, "label")
    return dataset_dict

def remove_unused_columns(dataset_dict):
    keep_cols = ['text', 'label', 'idx', 'id']
    for split_name, dataset in dataset_dict.items():
        dataset_dict[split_name] = dataset.remove_columns([c for c in dataset.features.keys() if c not in keep_cols])
    return dataset_dict

def remove_unused_columns_from_dataset(dataset):
    keep_cols = ['text', 'label', 'idx', 'id', 'features']
    dataset = dataset.remove_columns([c for c in dataset.features.keys() if c not in keep_cols])
    return dataset

class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs
    
    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)
    
    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()

def compare_policy_probs(policy_probs, augmented_dataset):
    transforms_applied = np.array(augmented_dataset["transforms_applied"])
    counts = transforms_applied.sum(axis=0)
    percents = counts / counts.sum()
    
    df = pd.DataFrame([policy_probs.mean(axis=0), percents]).T.fillna(0)
    df.columns = ["policy", "actual"]
    df["difference"] = df["policy"] - df["actual"]
    return df

def repeat_array(arr_a, arr_b):
    if len(arr_a) > len(arr_b):
        num_repeats = len(arr_a) // len(arr_b)
        remainder = len(arr_a) % len(arr_b)
        repeated_arr_b = np.tile(arr_b, num_repeats)
        repeated_arr_b = np.concatenate((repeated_arr_b, arr_b[:remainder]))
        return repeated_arr_b
    else:
        return arr_b
    
def count_duplicates(dataset):
    text_counts = Counter(dataset["text"])
    duplicate_counts = {text: count for text, count in text_counts.items() if count > 1}
    return len(duplicate_counts)

def replace_empty_text(dataset, replace_string = "<<empty>>"):
    dataset = dataset.map(
        lambda example: {**example, "text": replace_string} if example["text"] == "" or len(example["text"].strip().split()) == 0 else example
    )
    return dataset

def clear_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Failed to delete {dir_path}. Reason: {e}")