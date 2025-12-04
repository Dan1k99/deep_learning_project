import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_split_cifar10(batch_size=64, num_workers=2):
    """
    Downloads CIFAR-10 and splits it into two tasks.
    Task A: Labels 0-4
    Task B: Labels 5-9
    
    Returns:
        train_loader_A, test_loader_A, train_loader_B, test_loader_B
    """
    # Standard normalization for ResNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download datasets
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Helper to filter indices
    def get_task_indices(dataset, classes):
        targets = np.array(dataset.targets)
        indices = np.where(np.isin(targets, classes))[0]
        return indices

    # Define Classes
    task_a_classes = [0, 1, 2, 3, 4]
    task_b_classes = [5, 6, 7, 8, 9]

    # Create Subsets
    train_indices_a = get_task_indices(train_set, task_a_classes)
    test_indices_a = get_task_indices(test_set, task_a_classes)
    
    train_indices_b = get_task_indices(train_set, task_b_classes)
    test_indices_b = get_task_indices(test_set, task_b_classes)

    train_subset_a = Subset(train_set, train_indices_a)
    test_subset_a = Subset(test_set, test_indices_a)
    train_subset_b = Subset(train_set, train_indices_b)
    test_subset_b = Subset(test_set, test_indices_b)

    # Create Loaders [cite: 13-17]
    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': True}
    
    train_loader_a = DataLoader(train_subset_a, **loader_args)
    test_loader_a = DataLoader(test_subset_a, batch_size=batch_size, shuffle=False)
    
    train_loader_b = DataLoader(train_subset_b, **loader_args)
    test_loader_b = DataLoader(test_subset_b, batch_size=batch_size, shuffle=False)

    return train_loader_a, test_loader_a, train_loader_b, test_loader_b