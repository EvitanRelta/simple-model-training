# Simple Pytorch model training

This repository contains a simple script to train a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch.

It will download the MNIST dataset automatically if needed.

<br>

## Requirements

To run this script, you need to have PyTorch installed. For GPU acceleration, ensure you have CUDA installed on your system.

### Installing PyTorch

1. Visit one of the following websites:
   - For the latest version: https://pytorch.org/get-started/locally/#start-locally
   - For previous versions: https://pytorch.org/get-started/previous-versions/

2. When installing, select `CUDA` for the `Compute Platform` option to enable GPU support.

3. Follow the installation instructions provided on the website.

<br>

## Usage

Clone the repository:

```bash
git clone https://github.com/EvitanRelta/simple-model-training.git
cd simple-model-training
```

Then to train the model, simply run:

```bash
python train_mnist_model.py
```

The script will automatically use GPU if available, otherwise it will default to CPU.
