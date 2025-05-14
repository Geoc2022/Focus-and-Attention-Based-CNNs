import argparse

import attention_model
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import os
import numpy as np
from pathlib import Path

class ImageNet32Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True):
        self.data = []
        self.labels = []
        
        if train:
            for i in range(1, 11):
                file_path = os.path.join(data_dir, f'train_data_batch_{i}.npz')
                data_dict = np.load(file_path)
                images = data_dict['data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
                labels = data_dict['labels'] - 1
                self.data.append(images)
                self.labels.append(labels)
        else:
            file_path = os.path.join(data_dir, 'val_data.npz')
            data_dict = np.load(file_path)
            images = data_dict['data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            labels = data_dict['labels'] - 1
            self.data.append(images)
            self.labels.append(labels)
            
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # dataset1 = datasets.MNIST(
    #     "./../data", train=True, download=True, transform=transform
    # )
    # dataset2 = datasets.MNIST("./../data", train=False, transform=transform)
    # dataset1 = datasets.FashionMNIST(
    #     "./../data", train=True, download=True, transform=transform
    # )
    # dataset2 = datasets.FashionMNIST("./../data", train=False, transform=transform)

    #CIFAR10
    # dataset1 = datasets.CIFAR10(root="./../data", train=True, download=True, transform=transform)
    # dataset2 = datasets.CIFAR10(root="./../data", train=False, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #ImageNET
    train_dir = "./../data/Imagenet32_train_npz"
    val_dir = "./../data/Imagenet32_val_npz"

    train_dataset = ImageNet32Dataset(train_dir, train=True)
    val_dataset = ImageNet32Dataset(val_dir, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    model = attention_model.Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = attention_model.FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        attention_model.train(args, model, device, train_loader, optimizer, epoch, criterion)
        attention_model.test(model, device, test_loader, criterion)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
