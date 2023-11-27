import os
import subprocess
import yaml
import tempfile
from datasets import Dataset, concatenate_datasets

def create_huggingface_dataset_from_general_dataset(general_dataset_instance):
    # Extract texts and labels from the general_dataset instance
    texts = general_dataset_instance.texts
    labels = general_dataset_instance.labels

    # Prepare data in the format required by Hugging Face datasets
    data = {"text": texts, "label": labels}

    # Create and return a Hugging Face Dataset
    return Dataset.from_dict(data)

class TextAutoAugmenter:
    def __init__(self):
        self.repo_url = "https://github.com/lancopku/text-autoaugment.git"
        self.augmenter_directory = "./fada/augmenters/text_autoaugment"
        self.conda_env_name = "taa"
        self.python_version = "3.6"
        self.check_and_clone_repo()
        
    def run_command(self, command):
        """Run a shell command and handle exceptions."""
        try:
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")

    def check_and_clone_repo(self):
        if not os.path.exists(os.path.join(self.augmenter_directory)):
            print("Cloning Text-Autoaugment repository...")
            os.makedirs(self.augmenter_directory, exist_ok=True)
            subprocess.run(['git', 'clone', self.repo_url, self.augmenter_directory], check=True)
            os.chdir(self.augmenter_directory)

            # Install dependencies
            self.run_command("pip install git+https://github.com/wbaek/theconf")
            self.run_command("pip install git+https://github.com/ildoonet/pystopwatch2.git")
            self.run_command("pip install -r requirements.txt")

            # Install library
            self.run_command("python setup.py develop")

            # Download NLTK models
            self.run_command("python -c \"import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')\"")

    def __call__(self, dataset, name, save_path="./temp", num_aug=3):

        from fada.augmenters.text_autoaugment.taa.search_and_augment import augment_with_presearched_policy

        config_params = {
            'model': {'type': 'Bert'},
            'dataset': {
                'path': None,
                'name': name,
                'data_dir': save_path,
                'data_files': {'train': f"{name}.original.csv"},
                'text_key': 'text'
            },
            'abspath': self.augmenter_directory,
            'aug': 'name',
            'per_device_train_batch_size': 64,
            'per_device_eval_batch_size': 128,
            'epoch': 10,
            'lr': 4e-5,
            'max_seq_length': 128,
            'n_aug': num_aug,
            'num_op': 2,
            'num_policy': 4,
            'method': 'taa',
            'topN': 3,
            'ir': 1,
            'seed': 59,
            'trail': 1,
            'train': {'npc': 50},
            'valid': {'npc': 50},
            'test': {'npc': 50},
            'num_search': 200,
            'num_gpus': 4,
            'num_cpus': 40
        }

        # Create a temporary configuration file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.yaml') as temp_config_file:
            yaml.dump(config_params, temp_config_file, default_flow_style=False)
            config_file_path = temp_config_file.name

        # Perform augmentation
        augmented_dataset = augment_with_presearched_policy(dataset, configfile=config_file_path)
        augmented_dataset = create_huggingface_dataset_from_general_dataset(augmented_dataset)
        augmented_dataset = augmented_dataset.cast(dataset.features)
        augmented_dataset = concatenate_datasets([dataset, augmented_dataset])

        # Clean up the temporary file
        os.remove(config_file_path)

        return augmented_dataset

if __name__ == "__main__":
    from datasets import load_dataset
    
    augmenter = TextAutoAugmenter()

    dataset = load_dataset("imdb", split="train").select(range(1000))
    print(dataset)

    aug_dataset = augmenter(dataset, name="imdb", num_aug=3)
    print(aug_dataset)