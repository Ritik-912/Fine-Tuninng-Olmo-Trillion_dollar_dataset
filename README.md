# Fine-Tuninng-Olmo-Trillion_dollar_dataset

This repository contains code for fine-tuning the OLMo (Open Language Model) using the BitsAndBytes quantization technique and PEFT (Parameter Efficient Fine-Tuning) method. The goal is to fine-tune a causal language model for specific tasks related to monetary policy analysis.

## Prerequisites

Before running the code, make sure you have the following installed:

- Python 3.x
- PyTorch
- Transformers library from Hugging Face (`transformers`)
- Pandas
- OpenPyXL
- bitsandbytes (custom package for quantization)
- peft (custom package for parameter efficient fine-tuning)
- trl (Taming Transformers library for training)


## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/OLMo-Finetuning.git
   cd OLMo-Finetuning
   ```

2. Install the required packages:
   ```bash
   pip install -qU transformers accelerate bitsandbytes peft trl datasets evaluate ai2-olmo
   ```

3. Download the dataset files into a folder named data in same directory.
    ```bash
    # Create a new directory
    mkdir data
    cd data

    # Initialize Git
    git init

    # Set Up Remote
    git remote add -f origin https://github.com/gtfintechlab/fomc-hawkish-dovish

    # Enable Sparse Checkout
    git config core.sparsecheckout true

    # Specify Folder to Clone
    echo "training_data/test-and-training/*" >> .git/info/sparse-checkout

    # Pull Content
    git pull origin main
    ```

4. Run the main script:
   ```bash
   python olmo_fine_tuning.py
   ```

## Description

- `olmo_fine_tuning.py`: Main script containing code for model setup, data preparation, training, and evaluation.
- `bitsandbytes`: Custom package for BitsAndBytes quantization.
- `peft`: Custom package for Parameter Efficient Fine-Tuning.
- `trl`: Taming Transformers library for training.

## Files

- `README.md`: This file providing an overview of the repository.
- `data/training_data/`: Directory containing training and test data in Excel format.