import argparse

import saliency_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

patch_size = 5
k_points = 3

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Focus MNIST Example")
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
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
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
    parser.add_argument(
        "--show-patches",
        type=bool,
        default=True,
        help="Show patches and keypoints",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        # device = torch.device("mps")
        device = torch.device("cpu")
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
    dataset1 = datasets.MNIST("./../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./../data", train=False, transform=transform)
    # dataset1 = datasets.FashionMNIST("./../data", train=True, download=True, transform=transform)
    # dataset2 = datasets.FashionMNIST("./../data", train=False, transform=transform)
    # dataset1 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # dataset2 = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    image_shape = dataset1[0][0].shape
    model = saliency_model.KeypointPatchModel(k=k_points, patch_size=patch_size, image_shape=image_shape).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        saliency_model.train(args, model, device, train_loader, optimizer, criterion, epoch)
        saliency_model.test(model, device, test_loader, criterion)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    if args.show_patches:
        while input("Show patches and keypoints? (y/n): ").lower() != "n":
            saliency_model.show_patches_and_keypoints(model, device, test_loader)



if __name__ == "__main__":
    main()
