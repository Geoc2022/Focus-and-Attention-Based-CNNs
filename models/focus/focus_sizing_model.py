import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.optim as optim

import matplotlib.pyplot as plt
import random


def create_patch_grid(k, patch_size):
    lin_coords = torch.linspace(-1, 1, steps=patch_size)
    x_grid, y_grid = torch.meshgrid(lin_coords, lin_coords, indexing="xy")
    grid = torch.stack([x_grid, y_grid], dim=-1)
    grid = grid.unsqueeze(0).repeat(k, 1, 1, 1)
    return grid

class KeypointDetector(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, 2 * k)

    def forward(self, x):
        features = self.cnn(x)
        coords = self.fc(features.view(x.size(0), -1))
        coords = coords.view(-1, self.k, 2)
        coords = coords.clamp(-1, 1)
        return coords

class PatchClassifier(nn.Module):
    def __init__(self, k, patch_size, num_classes=10):
        super().__init__()
        self.k = k
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * patch_size * patch_size, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64 * k, num_classes)

    def forward(self, patches):
        B, k, C, H, W = patches.shape
        patches = patches.view(B * k, C, H, W)
        features = self.cnn(patches)
        features = features.view(B, k * 64)
        out = self.classifier(features)
        return out

class KeypointPatchModel(nn.Module):
    def __init__(self, k=3, patch_size=5, image_size=28):
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.image_size = image_size
        self.detector = KeypointDetector(k)
        self.classifier = PatchClassifier(k, patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        coords = self.detector(x)
        grid = create_patch_grid(self.k, self.patch_size).to(x.device)
        coords = coords.unsqueeze(2).unsqueeze(2)
        patch_grid = grid.unsqueeze(0) + coords
        patch_grid = patch_grid.view(B * self.k, self.patch_size, self.patch_size, 2)
        x_rep = x.unsqueeze(1).repeat(1, self.k, 1, 1, 1)
        x_rep = x_rep.view(B * self.k, C, H, W)
        patches = F.grid_sample(x_rep, patch_grid, align_corners=True)
        patches = patches.view(B, self.k, C, self.patch_size, self.patch_size)
        return self.classifier(patches)
    
def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    accuracy = 100. * correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def show_patches_and_keypoints(model, device, loader):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        keypoints = model.detector(images)  
        vis_keypoints = keypoints.clone()   

        B, C, H, W = images.shape
        grid = create_patch_grid(model.k, model.patch_size).to(device)
        keypoints = keypoints.unsqueeze(2).unsqueeze(2)  
        patch_grid = grid.unsqueeze(0) + keypoints
        patch_grid = patch_grid.view(B * model.k, model.patch_size, model.patch_size, 2)
        images_rep = images.unsqueeze(1).repeat(1, model.k, 1, 1, 1)
        images_rep = images_rep.view(B * model.k, C, H, W)
        patches = F.grid_sample(images_rep, patch_grid, align_corners=True)
        patches = patches.view(B, model.k, C, model.patch_size, model.patch_size)

    idx = random.randint(0, B - 1)
    fig, axes = plt.subplots(1, model.k + 1, figsize=(15, 5))
    axes[0].imshow(images[idx].cpu().permute(1, 2, 0))
    axes[0].scatter(
        (vis_keypoints[idx, :, 0].cpu().numpy() + 1) * model.image_size / 2,
        (vis_keypoints[idx, :, 1].cpu().numpy() + 1) * model.image_size / 2,
        c="red",
        s=50,
        marker="x"
    )
    axes[0].set_title("Original Image with Keypoints")

    for i in range(model.k):
        axes[i + 1].imshow(patches[idx, i].cpu().permute(1, 2, 0))
        axes[i + 1].set_title(f"Patch {i + 1}")

    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        output = model(images[idx].unsqueeze(0))
        pred = output.argmax(1).item()
        print(f"Predicted label: {pred} \t True label: {labels[idx].item()} \t Correct: {pred == labels[idx].item()}")

