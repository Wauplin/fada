

class TrainPipeline:
    
    def __init__(self, 
                 model_id="bert-base-cased", 
                 dataset_config=("glue", "sst2"),
                 train_batch_size = 8,
                 eval_batch_size = 16,
                 gradient_accumulation_steps = 1,
                 num_epoch = 10,
                 seed=130):
        self.model_id = model_id
        self.dataset_config = dataset_config
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epoch = num_epoch
        self.seed = seed
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = "./tmp_TrainPipeline/"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # initializations =========================================================
        self.prepare_valtest()
        
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding=True, truncation=True, max_length=250)
        
    def prepare_valtest(self):
        if 'sst2' in self.dataset_config:
            val_dataset  = load_dataset(*self.dataset_config, split='validation')
            val_dataset  = val_dataset.rename_column("sentence", "text")
            test_val     = val_dataset.train_test_split(test_size=0.5)
            val_dataset  = test_val['train']
            test_dataset = test_val['test']
            
        self.num_classes = test_dataset.features['label'].num_classes

        val_dataset = val_dataset.map(self.tokenize_function, batched=True, batch_size=1000)
        val_dataset = val_dataset.rename_column("label", "labels") 
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        test_dataset = test_dataset.map(self.tokenize_function, batched=True, batch_size=1000)
        test_dataset = test_dataset.rename_column("label", "labels") 
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
            
    def train(self, train_dataset):
        
        #############################################################
        ## Model + Tokenizer ########################################
        #############################################################
        
        checkpoint = self.save_dir + self.model_id + '-' + "_".join(self.dataset_config)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_classes).to(self.device)
        
        #############################################################
        ## Dataset Preparation ######################################
        #############################################################
        
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=1000)
        train_dataset = train_dataset.rename_column("label", "labels") 
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        self.train_dataset = train_dataset
    
        #############################################################
        ## Callbacks ################################################
        #############################################################
        
        callbacks = []

        escb = EarlyStoppingCallback(
            early_stopping_patience=3
        )
        callbacks.append(escb)
        
        #############################################################
        ## Training  ################################################
        #############################################################
        
        max_steps = int((len(train_dataset) * self.num_epoch / self.gradient_accumulation_steps) / self.train_batch_size)
        logging_steps = max_steps // self.num_epoch
        
        training_args = TrainingArguments(
            output_dir=checkpoint,
            overwrite_output_dir=True,
            max_steps=max_steps,
            save_steps=logging_steps,
            save_total_limit=1,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps, 
            warmup_steps=int(max_steps / 10),
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=logging_steps,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            evaluation_strategy="steps",
            remove_unused_columns=False
        )
        
        self.trainer = Trainer(
            model=self.model, 
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,                  
            train_dataset=self.train_dataset,         
            eval_dataset=self.val_dataset,
            callbacks=callbacks
        )
        
        self.trainer.train()
        
        shutil.rmtree(self.save_dir)
        
    def calculate_performance(self):
        self.trainer.eval_dataset = self.test_dataset
        return self.trainer.evaluate()
    
    def calculate_accuracy(self):
        perf = self.calculate_performance()
        return perf["eval_accuracy"]
    
    def evaluate(self, dataset):
        self.train(dataset) 
        return self.calculate_accuracy()