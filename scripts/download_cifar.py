import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar_dataloaders(batch_size=64, num_workers=2, data_dir="./data"):
    # CIFAR-10 mean and std normalization constants
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    print("Downloading CIFAR-10 and testing DataLoader setup...")
    train_dl, test_dl = get_cifar_dataloaders(batch_size=64)
    xb, yb = next(iter(train_dl))
    print(f"Dataset downloaded successfully! Batch shape: {xb.shape}, Labels shape: {yb.shape}")