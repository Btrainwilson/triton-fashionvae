import torch
from torchvision import datasets, transforms

def download_fashion_mnist(root_dir='data', device='cpu'):
    """
    Download and transform the FashionMNIST dataset.

    Parameters:
    root_dir (str): Directory where the dataset will be stored.
    """

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download the training data
    train_set = datasets.FashionMNIST(root=root_dir, train=True, download=True, transform=transform)

    # Download the test data
    test_set = datasets.FashionMNIST(root=root_dir, train=False, download=True, transform=transform)

    return train_set, test_set

