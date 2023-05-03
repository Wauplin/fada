from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_callback import TrainerControl
from datasets import load_dataset, load_metric, load_from_disk
import os
import importlib
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

import textattack
from textattack import Attacker
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import (
    TextFoolerJin2019, 
    DeepWordBugGao2018, 
    Pruthi2019, 
    TextBuggerLi2018, 
    PSOZang2020, 
    CheckList2020, 
    BERTAttackLi2020
)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from sibyl import *
from fada.utils import *

# https://www.gitmemory.com/issue/QData/TextAttack/424/795806095

def load_class(module_class_str):
    parts = module_class_str.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        out = []
        i = 0
        while i < len(text_input_list):
            batch = text_input_list[i : i + self.batch_size]
            encoding = self.tokenizer(batch, padding=True, truncation=True, max_length=250, return_tensors='pt')
            outputs = self.model(encoding['input_ids'].to(self.device), attention_mask=encoding['attention_mask'].to(self.device))
            preds = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu()
            out.append(preds)
            i += self.batch_size
        out = torch.cat(out)
        return out

parser = argparse.ArgumentParser(description='Sibyl Adversarial Attacker')

parser.add_argument('--num_runs', default=3, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--dataset-config', nargs='+', default=['glue', 'sst2'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-keys', nargs='+', default=['text'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--models', nargs='+',  default=['bert-base-uncased'], 
                    type=str, help='pretrained huggingface models to attack')
parser.add_argument('--attacks', 
                    nargs='+',  
                    default=[
                        'textattack.attack_recipes.TextFoolerJin2019',
                        'textattack.attack_recipes.DeepWordBugGao2018',
                        'textattack.attack_recipes.TextBuggerLi2018',
                        'textattack.attack_recipes.PSOZang2020',
                        'textattack.attack_recipes.CheckList2020',
                        'textattack.attack_recipes.BERTAttackLi2020',
                    ], 
                    type=str, help='pretrained huggingface models to attack')
parser.add_argument('--save-path', type=str, default='NLP_adv_robustness.csv',
                    help='name for the csv file to save with results')
parser.add_argument('--num_advs', default=100, type=int, metavar='N',
                    help='number of adversarial examples to generate')

args = parser.parse_args()

def robustness(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    recipes = [load_class(a) for a in args.attacks]

    run_args = []
    for run_num in range(args.num_runs):
            for model in args.models:
                run_args.append({
                    "run_num":run_num,
                    "model":model,
                })

    results = []
    save_path = args.save_path  
    if os.path.exists(save_path):
        results.extend(pd.read_csv(save_path).to_dict("records"))
        start_position = len(results)
    else:
        start_position = 0

    print('starting at position {}'.format(start_position))
    for run_arg in run_args[start_position:]:
        
        #############################################################
        ## Initializations ##########################################
        #############################################################
        run_num = run_arg['run_num']
        model_checkpoint = run_arg['model']

        print(pd.DataFrame([run_arg]))

        if len(args.dataset_keys) == 1:
            sentence1_key, sentence2_key = args.dataset_keys[0], None
        else:
            # if not 1 then assume 2 keys
            sentence1_key, sentence2_key = args.dataset_keys

        #############################################################
        ## Dataset Preparation ######################################
        #############################################################

        dataset_name = args.dataset_config[0]

        raw_datasets = load_dataset(*args.dataset_config)
        if 'sst2' in args.dataset_config:
            raw_datasets.pop("test") # test set is not usable (all labels -1)

        raw_datasets = prepare_splits(raw_datasets)
        raw_datasets = rename_text_columns(raw_datasets)
        raw_datasets = raw_datasets.shuffle(seed=run_num)

        dataset = raw_datasets["test"]
        num_labels = len(dataset.features["label"].names)

        print(dataset)
        print('Number of classes:', num_labels)
                
        #############################################################
        ## Model + Tokenizer ########################################
        #############################################################

        if os.path.exists(model):
            recent_checkpoint = [name for name in os.listdir(checkpoint) if 'checkpoint' in name]
            if recent_checkpoint:
                checkpoint = os.path.join(checkpoint, recent_checkpoint[-1])
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

        #############################################################
        ## TextAttacks ##############################################
        #############################################################

        out = {}
        out.update({
            "run_num":          run_num,
            "checkpoint":       checkpoint,
            "dataset_config":   args.dataset_config,
            "num_adversaries":  args.num_adversaries,
        })

        if loaded_checkpoint:

            mw = CustomModelWrapper(model, tokenizer)
            dataset = HuggingFaceDataset(test_dataset, shuffle=True)
            attack_args = textattack.AttackArgs(num_examples=args.num_advs, disable_stdout=True)

            for recipe in recipes:

                attack = recipe.build(mw)
                attacker = Attacker(attack, dataset, attack_args)
                attack_results = attacker.attack_dataset()

                num_results = 0
                num_failures = 0
                num_successes = 0

                for result in attack_results:                
                    num_results += 1
                    if (type(result) == textattack.attack_results.SuccessfulAttackResult or 
                        type(result) == textattack.attack_results.MaximizedAttackResult):
                        num_successes += 1
                    if type(result) == textattack.attack_results.FailedAttackResult:
                        num_failures += 1

                attack_success = num_successes / num_results
                out['attack_success_' + recipe.__name__] = attack_success

                print("{0} Attack Success: {1:0.2f}".format(recipe.__name__, attack_success))

        results.append(out)

        # save results
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    robustness(args)