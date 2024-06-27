from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset


def main() -> None:
    """Main function to orchestrate the MNIST download, model creation, training, and inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset = download_mnist()
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    model = create_model()
    model.to(device)  # Move model to GPU/CPU
    train_model(model, train_loader, device=device)
    run_inference(model, test_loader, device=device)


def download_mnist() -> Tuple[VisionDataset, VisionDataset]:
    """Downloads MNIST dataset and returns train and test datasets."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: VisionDataset, test_dataset: VisionDataset
) -> Tuple[DataLoader[Tuple[Tensor, Tensor]], DataLoader[Tuple[Tensor, Tensor]]]:
    """Creates and returns DataLoaders for train and test datasets."""
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def create_model() -> nn.Sequential:
    """Creates and returns a simple CNN model using nn.Sequential."""
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(1600, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model


def train_model(
    model: nn.Sequential,
    train_loader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device,
    num_epochs: int = 5,
) -> None:
    """Trains the model on the training data for a specified number of epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output: Tensor = model.forward(data)
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

def run_inference(
    model: nn.Sequential,
    test_loader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device,
) -> None:
    """Runs inference on the test data and prints the accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
