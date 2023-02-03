import numpy as np
from sibyl import acc_at_k
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy_score(labels, predictions.argmax(-1))
    precision, recall, fbeta_score, support = precision_recall_fscore_support(
        y_true=labels, 
        y_pred=predictions.argmax(-1), 
        average="weighted", 
        zero_division=0)
    return { 'accuracy': acc , 
             'precision': precision, 
             'recall': recall, 
             'fbeta_score': fbeta_score} 

def compute_accuracy(predictions, labels):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if len(labels.shape) > 1:
        acc = acc_at_k(labels, predictions, k=2)       
    else:
        acc = accuracy_score(labels, np.argmax(predictions, -1))
    return acc

def vectorize(output):
    sorted_output = sorted(output, key=lambda d: d['label']) 
    probs = np.array([d['score'] for d in sorted_output])
    return probs

def sample_transforms(transforms, p, n=2, replace=False):
    return np.random.choice(transforms, size=n, p=p, replace=replace).tolist()

def prepare_splits(dataset_dict, train_val_split = 0.9, val_test_split = 0.5):
    has_train = has_val = has_test = False
    train_id, val_id, test_id = "train", "valid", "test"
    for split_name in dataset_dict.keys():
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
        val_test      = dataset_dict[val_id].train_test_split(train_size=val_test_split)
        train_dataset = dataset_dict[train_id]
        val_dataset   = val_test['train']
        test_dataset  = val_test['test']
    if has_test and not has_val:
        train_val     = dataset_dict[train_id].train_test_split(train_size=train_val_split)
        train_dataset = train_val['train']
        val_dataset   = train_val['test']
        test_dataset  = dataset_dict[test_id]
    if not has_val and not has_test:
        train_val     = dataset_dict[train_id].train_test_split(train_size=train_val_split)
        val_test      = train_val['test'].train_test_split(train_size=val_test_split)
        train_dataset = train_val['train']
        val_dataset   = val_test['train']
        test_dataset  = val_test['test']

    dataset_dict[train_id] = train_dataset
    dataset_dict[val_id]   = val_dataset
    dataset_dict[test_id]  = test_dataset

    return dataset_dict

def rename_text_columns(dataset_dict):
    text_columns = ["sentence"]
    val_columns = ["val", "valid"]
    for split_name, dataset in dataset_dict.items():
        for column in dataset.features:
            if column in text_columns:
                dataset_dict[split_name] = dataset.rename_column(column, "text")
            if column in val_columns:
                dataset_dict[split_name] = dataset.rename_column(column, "validation")
    return dataset_dict

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