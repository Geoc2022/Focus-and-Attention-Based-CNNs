import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random


def extract_patches(images, keypoints, patch_size, device):
    B, C, H, W = images.shape
    k = keypoints.shape[1]

    coords_normalized = (keypoints + 1) / 2

    patch_radius = patch_size / H
    patch_grid = torch.linspace(-patch_radius, patch_radius, patch_size, device=device)
    yy, xx = torch.meshgrid(patch_grid, patch_grid, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).unsqueeze(0)

    coords_expanded = coords_normalized.unsqueeze(2).unsqueeze(3)
    grid = grid + coords_expanded

    grid = grid * 2 - 1
    grid = grid.view(B * k, patch_size, patch_size, 2)

    x_expanded = images.repeat_interleave(k, dim=0)
    patches = F.grid_sample(x_expanded, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

    patches = patches.view(B, k, C, patch_size, patch_size)
    return patches


class KeypointDetector(nn.Module):
    def __init__(self, k, channel_size):
        super().__init__()
        self.k = k
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 16, 3, padding=1),
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
    def __init__(self, k, patch_size, channel_size, num_classes):
        super().__init__()
        self.k = k
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 8, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * patch_size * patch_size, 64),
            nn.ReLU()
        )
        # self.classifier = nn.Linear(64 * k, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(.25),
            nn.Linear(64 * k, 64 * k),
            nn.Dropout(.5),
            nn.Linear(64 * k, num_classes)
        )

    def forward(self, patches):
        B, k, C, H, W = patches.shape
        patches = patches.view(B * k, C, H, W)
        features = self.cnn(patches)
        features = features.view(B, k * 64)
        out = self.classifier(features)
        return out

class KeypointPatchModel(nn.Module):
    def __init__(self, k=3, patch_size=5, image_shape=(28, 28, 1), num_classes=10):
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.image_shape = image_shape
        self.detector = KeypointDetector(
            k=k, 
            channel_size=image_shape[0]
        )
        self.classifier = PatchClassifier(
            k=k, 
            patch_size=patch_size, 
            channel_size=image_shape[0], 
            num_classes=num_classes
        )

    def forward(self, x):
        coords = self.detector(x)

        patches = extract_patches(x, coords, self.patch_size, x.device)
                
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
    return correct / len(test_loader.dataset), test_loss


def show_patches_and_keypoints(model, device, loader):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        keypoints = model.detector(images)  
        vis_keypoints = keypoints.clone()   
        B, C, H, W = images.shape

        patches = extract_patches(images, keypoints, model.patch_size, device)

    idx = random.randint(0, B - 1)
    fig, axes = plt.subplots(1, model.k + 1, figsize=(15, 5))
    axes[0].imshow(images[idx].cpu().permute(1, 2, 0))
    axes[0].scatter(
        (vis_keypoints[idx, :, 0].cpu().numpy() + 1) * model.image_shape[1] / 2,
        (vis_keypoints[idx, :, 1].cpu().numpy() + 1) * model.image_shape[1] / 2,
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

