import torch
import numpy as np
from utils import *
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter

class TestPipeline:
    def __init__(self):
        self.api = HfApi()
        self.pipe = None
        self.device = 0 if torch.cuda.is_available() else -1

    def find_model_for_dataset(self, dataset_name):
        
        model_filter = ModelFilter(
            task="text-classification",
            library="pytorch",
            # model_name=dataset_name,
            trained_dataset=dataset_name)

        model_id = next(iter(self.api.list_models(filter=model_filter)))

        if model_id:
            model_id = getattr(model_id, 'modelId')
            print('Using ' + model_id + ' to support evaluation.')
            self.pipe = pipeline("text-classification", 
                                 model=model_id, 
                                 device=self.device, 
                                 padding=True, 
                                 truncation=True,
                                 top_k=None)

    def extract_prediction_probabilities(self, dataset):
        output = self.pipe(dataset['text'])
        return np.stack([vectorize(o) for o in output])
    
    def extract_prediction_classes(self, dataset):
        return np.argmax(self.extract_prediction_probabilities(dataset), axis=1)
    
    def calculate_accuracy(self, dataset):
        preds = self.extract_prediction_probabilities(dataset)
        acc = compute_accuracy(preds, dataset["label"])
        return acc
    
    def evaluate(self, dataset):
        return self.calculate_accuracy(dataset)