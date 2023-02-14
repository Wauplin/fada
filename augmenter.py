from utils import sample_transforms

class Augmenter:
    def __init__(self, 
                 dataset,
                 transforms,
                 transform_probabilities, 
                 num_augmentations_per_record = 5,
                 num_transforms_to_apply = 2,
                 batch_size = 10,
                 keep_originals = False,
                 allow_resampling = False):
        
        self.dataset = dataset
        self.transforms = transforms
        self.transform_probabilities = transform_probabilities
        self.num_augmentations_per_record = num_augmentations_per_record
        self.num_transforms_to_apply = num_transforms_to_apply
        self.batch_size = batch_size
        self.keep_originals = keep_originals
        self.allow_resampling = allow_resampling
        
        # initializations
        self.add_idx_to_dataset()
        
    def add_idx_to_dataset(self):
        if 'idx' not in self.dataset.features:
            self.dataset = self.dataset.add_column("idx", range(len(self.dataset)))
        
    def apply_to_batch(self, batch):
        new_texts, new_labels = [], []
        for idx, text, label in zip(batch['idx'], batch['text'], batch['label']):
            for _ in range(self.num_augmentations_per_record):
                sampled_transforms = sample_transforms(self.transforms, 
                                                       p=self.transform_probabilities[idx], 
                                                       n=self.num_transforms_to_apply, 
                                                       replace=self.allow_resampling)
                for t in sampled_transforms:
                    try:
                        text, label = t.apply([text], [label])
                        text, label = text[0], label[0]
                    except Exception as e: 
                        print(e)
                        print("[Augmenter]: skipping augmentation and continuing on...")
                new_texts.append(text)
                new_labels.append(label)
                
        if self.keep_originals:
            out_texts = batch['text'] + new_texts
            out_labels = batch['label'] + new_labels
            return {
                "text": out_texts, 
                "label": out_labels,
                "idx": list(range(len(out_texts)))
            }
        else:
            return {
                "text": new_texts, 
                "label": new_labels, 
                "idx": list(range(len(new_texts)))}
                                                   
    def augment(self):
        return self.dataset.map(self.apply_to_batch, batched=True, batch_size=self.batch_size)