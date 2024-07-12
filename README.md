# Description

Briefly describe your project here. Mention what it does, key features, and why it is useful.

# Installation Guide

## Step 1: Install Conda

Follow the instructions at the [official Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Anaconda Distribution or Miniconda on your system. Make sure to install the version that supports Python 3.8 or higher

## Step 2: Create and Activate Conda Environment

Open your terminal or command prompt and execute the following commands:

```sh
conda create -n test
conda activate test
```
## Step 3: Install Required Libraries

```sh
conda install numpy pandas matplotlib seaborn xgboost tqdm scikit-learn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```