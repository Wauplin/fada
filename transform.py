import torch
import inspect
import random

class Transform:
    def __init__(self, transform_class, num_outputs=1, task_name="sentiment"):
        self.transform_class = transform_class
        self.num_outputs = num_outputs
        self.task_name = task_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.intakes_target = False
        self.is_batched = False
        
        # setting class attributes
        if 'to_tense' in inspect.signature(self.transform_class).parameters:
            print("initializing class with to_tense='past'") # future & random don't work
            self.transform_instance = self.transform_class(to_tense="past")
        elif 'source_lang' in inspect.signature(self.transform_class).parameters:
            print("initializing class with source_lang='es'") 
            self.transform_instance = self.transform_class(source_lang="es")
        elif 'task_name' in inspect.signature(self.transform_class).parameters:
            print(f"initializing class with task_name='{task_name}', return_metadata=True") 
            self.transform_instance = self.transform_class(task_name=self.task_name, return_metadata=True)
        # elif isinstance(self.transform_class, LostInTranslation):
        #     self.transform_instance = self.transform_class(device=0)
        else:
            self.transform_instance = self.transform_class()
        
        # setting instance attributes
        if hasattr(self.transform_instance, "max_outputs"):
            print(f"setting max_outputs={self.num_outputs}")
            self.transform_instance.max_outputs = self.num_outputs
        if hasattr(self.transform_instance, "max_paraphrases"):
            print(f"setting max_paraphrases={self.num_outputs}")
            self.transform_instance.max_paraphrases = self.num_outputs
        if hasattr(self.transform_instance, "device"):
            if self.transform_instance.device is None or self.transform_instance.device == 'cpu':
                print(f"setting device={self.device}")
                self.transform_instance.device = self.device
        
        # selecting the transformation function
        if hasattr(self.transform_class, "generate"):
            self.transform_fn = self.transform_instance.generate
        if hasattr(self.transform_class, "augment"):
            self.transform_fn = self.transform_instance.augment
        if hasattr(self.transform_class, "transform_batch"):
            self.transform_fn = self.transform_instance.transform_batch
            self.intakes_target = True
            self.is_batched = True
            
    def synced_shuffle(self, list1, list2):
        # Shuffle two lists with same order
        # Using zip() + * operator + shuffle()
        temp = list(zip(list1, list2))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        res1, res2 = list(res1), list(res2)
        return res1, res2
            
    def apply(self, texts, labels=None):
        if self.intakes_target:
            if self.is_batched:
                new_texts, new_labels = self.transform_fn((texts, labels))
            else:
                new_texts, new_labels = [], []
                for t, l in zip(texts, labels):
                    new_t, new_l = self.transform_fn(t, l)
                    new_texts.append(new_t)
                    new_labels.extend([new_l] * len(new_t))
        else:
            if self.is_batched:
                new_texts = self.transform_fn((texts))
                new_texts = labels
            else:
                new_texts, new_labels = [], []
                for t, l in zip(texts, labels):
                    new_t = self.transform_fn(t)
                    if len(new_t) > self.num_outputs:
                        new_t = new_t[:self.num_outputs]
                    new_texts.extend(new_t)
                    new_labels.extend([l] * len(new_t))
                    
        # post processing since some transformations add/remove more new outputs than expected
        if len(new_texts) == 0:
            print("no new_texts, substituting original texts...")
            new_texts = texts
        if len(new_labels) == 0:
            print("no new_labels, substituting original labels...")
            new_labels = labels
        new_texts, new_labels = self.synced_shuffle(new_texts, new_labels)
        
        expected_len = len(texts) * self.num_outputs
        new_texts = new_texts[:expected_len]
        new_labels = new_labels[:expected_len]
        
        return new_texts, new_labels