import os
import subprocess
import csv
from datasets import load_dataset

class EDANLPAugmentor:
    def __init__(self):
        self.repo_url = "https://github.com/jasonwei20/eda_nlp.git"
        self.augmenter_directory = "./fada/augmenters/eda_nlp"
        self.check_and_clone_repo()

    def check_and_clone_repo(self):
        if not os.path.exists(os.path.join(self.augmenter_directory)):
            print("Cloning EDA NLP repository...")
            os.makedirs(self.augmenter_directory, exist_ok=True)
            subprocess.run(['git', 'clone', self.repo_url, self.augmenter_directory], check=True)

    def hf_dataset_to_tsv(self, dataset, save_path=None):
        if not save_path:
            save_path = os.path.join(self.augmenter_directory,
                                     f"{dataset.builder_name}.{dataset.config_name}.temp.original.tsv".replace("/", "."))
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in dataset:
                writer.writerow([row['label'], row['text']])
        return save_path

    def run_augmentation_script(self, input_filename, num_aug=3):
        save_path = f"{input_filename.replace('original', 'eda')}"
        try:
            command = ['python', 
                       f'{self.augmenter_directory}/code/augment.py', 
                       f'--input={input_filename}', 
                       f'--num_aug={num_aug}', 
                       f'--output={save_path}']

            print(command)
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("STDOUT:", result.stdout.decode())
            print("STDERR:", result.stderr.decode())
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return save_path

    def __call__(self, dataset, num_aug=3):
        tsv_path = self.hf_dataset_to_tsv(dataset)
        aug_path = self.run_augmentation_script(tsv_path, num_aug)
        aug_dataset = load_dataset('csv', data_files=aug_path, delimiter='\t', column_names=['label', 'text'])["train"]
        os.remove(tsv_path); os.remove(aug_path)
        return aug_dataset

if __name__ == "__main__":
    augmenter = EDANLPAugmentor()

    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.rename_column("sentence", "text")
    print(dataset)

    aug_dataset = augmenter(dataset, num_aug=3)
    print(aug_dataset)