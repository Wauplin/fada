import os
import argparse

import torch
from augmenter import Augmenter
from datasets import load_dataset, concatenate_datasets

# argparse

parser = argparse.ArgumentParser(description='TextDiversity Paraphrase Dataset Generator')

parser.add_argument('--dataset-config', nargs='+', default=['glue', 'sst2'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-keys', nargs='+', default=['text'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--data-dir', type=str, default="./datasets/",
                    help='directory in which to save the augmented data')
parser.add_argument('--num_augmentations_per_record', default=1, type=int, metavar='N',
                    help='augmentation multiplier - number of new inputs per 1 original')
parser.add_argument('--num_transforms_to_apply', default=1, type=int, metavar='N',
                    help='number of augmentations to sample and apply to each input')
parser.add_argument('--batch-size', default=10, type=int, metavar='N',
                    help='number of inputs to proccess per iteration')
parser.add_argument('--techniques', nargs='+', 
                    default=['fada', 'uniform'],
                    type=str, help='technique used to generate the augmented datasets')
parser.add_argument('--keep-original', default=True, action='store_true',
                    help='preserve original dataset in the updated one')
parser.add_argument('--force', default=False, action='store_true',
                    help='force the dataset creation even if one already exists')

args = parser.parse_args()

# load dataset and process
dataset = load_dataset(*args.dataset_config, split='train') 

# standardizing dataset column names
if 'qqp' in args.dataset_config:
    dataset = dataset.rename_column("question1", "sentence1")
    dataset = dataset.rename_column("question2", "sentence2")
if 'sst2' in args.dataset_config:
    dataset = dataset.rename_column("sentence", "text")

key_columns = ['text', 'sentence1', 'sentence2', 'label']
columns_to_remove = [c for c in dataset.column_names if c not in key_columns]
dataset = dataset.remove_columns(columns_to_remove)

original_dataset_len = len(dataset)
print('original_dataset_len:', original_dataset_len)

# generate updated datasets for each paraphrase technique
for technique in args.techniques:

    print(f'Running {technique}')

    save_name = "_".join(args.dataset_config) + "_" + technique
    save_path = os.path.join(args.data_dir, save_name)

    if os.path.exists(save_path) and not args.force:
        print(f"existing dataset found at {save_path}. skipping...")
        continue

    # load paraphrase augmentation technique
    augmenter = Augmenter(
        dataset=fada.dataset, 
        transforms=fada.transforms, 
        transform_probabilities=uniform_policy, 
        num_augmentations_per_record = 5,
        num_transforms_to_apply = 2,
        keep_originals=args.keep_originals)

    # augment dataset
    updated_dataset = dataset.map(augmenter, 
                                  batched=True, 
                                  batch_size=args.batch_size)

    updated_dataset_len = len(updated_dataset)
    print('updated_dataset_len:', updated_dataset_len)

    # check for expected number of data points
    assert original_dataset_len * args.num_outputs == updated_dataset_len

    # check for new data actually being paraphrases (not duplicates)
    assert len(set(updated_dataset[-args.num_outputs:][args.dataset_keys[0]])) > 1 

    if args.keep_original:
        updated_dataset = concatenate_datasets([dataset, updated_dataset])

    # save updated dataset
    updated_dataset.save_to_disk(save_path)

    # print examples of updated dataset just to be sure they're as expected
    num_to_check = args.num_outputs * 3
    print('example:', updated_dataset[-num_to_check:])

    torch.cuda.empty_cache()