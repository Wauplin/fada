import os
import subprocess

class TextAutoAugmenter:
    def __init__(self):
        self.repo_url = "https://github.com/lancopku/text-autoaugment.git"
        self.augmenter_directory = "./fada/augmenters/text_autoaugment"
        self.conda_env_name = "taa"
        self.python_version = "3.6"
        self.check_and_clone_repo()

        print(os.environ['CONDA_DEFAULT_ENV'])

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

            # Create and activate Conda environment
            self.run_command(f"conda create -n {self.conda_env_name} python={self.python_version} -y")
            self.run_command(f"conda activate {self.conda_env_name}")

            # Install dependencies
            self.run_command("pip install git+https://github.com/wbaek/theconf")
            self.run_command("pip install git+https://github.com/ildoonet/pystopwatch2.git")
            self.run_command("pip install -r requirements.txt")

            # Install library
            self.run_command("python setup.py develop")

            # Download NLTK models
            self.run_command("python -c \"import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')\"")

    def __call__(self, dataset, name=None, num_aug=3):
        from fada.augmenters.text_autoaugment.taa.data import augment
        from fada.augmenters.text_autoaugment.taa.archive import policy_map

        """Augment dataset using presearched policy with an optional config."""
        self.run_command(f"conda activate {self.conda_env_name}")

        assert name in list(policy_map.keys()), f"No policy was found for this dataset. Must be one of {','.join(list(policy_map.keys()))}"
        policy = policy_map[name]
        augmented_train_dataset = augment(dataset=dataset, policy=policy, n_aug=num_aug)
        
        self.run_command(f"conda deactivate")
        self.run_command(f"conda activate fada")

        return augmented_dataset

# Usage

if __name__ == "__main__":
    from datasets import load_dataset
    
    augmenter = TextAutoAugmenter()

    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.rename_column("sentence", "text")
    print(dataset)

    aug_dataset = augmenter(dataset, name="sst5", num_aug=3)
    print(aug_dataset)