import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

import matplotlib.pyplot as plt
import random

def harris_corner_detection_coords(images, k, patch_size=5):
    # given images are in the shape of (B, C, H, W) find the keypoints using Harris corner detection
    
    # # Method1
    # B, C, H, W = images.shape
    # images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (B, H, W, C)
    # if C != 1:
    #     images_np = np.stack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images_np], axis=0)

    # gray_images = np.float32(images_np)
    # margin = 5  # should change to the patch size
    # mask = np.zeros((H, W), dtype=np.uint8)
    # mask[margin:H-margin, margin:W-margin] = 1

    # dst = np.array([cv2.cornerHarris(img, 2, 7, 0.04) for img in gray_images])
    # dst = np.array([cv2.dilate(d, None) for d in dst])
    # dst = dst * mask[None, :, :]

    # coords = []
    # for i in range(B):
    #     coords_i = np.argwhere(dst[i] > 0.01 * dst[i].max())
    #     coords_i = coords_i[:k]
    #     coords_i = (coords_i / np.array([H, W]) - 0.5) * 2
    #     coords.append(coords_i)

    # coords = np.stack(coords)
    # coords = torch.tensor(coords, dtype=torch.float32, device=images.device)
    # coords = coords.view(B, k, 2)
    # coords = coords.permute(0, 2, 1)
    # coords = coords.clamp(-1, 1)
    
    # return coords

    # Method2
    B, C, H, W = images.shape
    margin = patch_size
    mask = torch.zeros((H, W), device=images.device, dtype=torch.float32)
    mask[margin:H-margin, margin:W-margin] = 1

    if C != 1:
        images = torch.mean(images, dim=1, keepdim=True)

    images = images.squeeze(1)
    dx = F.conv2d(images.unsqueeze(1), torch.tensor([[[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]], device=images.device, dtype=images.dtype), padding=1)
    dy = F.conv2d(images.unsqueeze(1), torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], device=images.device, dtype=images.dtype), padding=1)

    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=images.device, dtype=images.dtype) / 16.0
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    Sxx = F.conv2d(Ixx, kernel, padding=1)
    Syy = F.conv2d(Iyy, kernel, padding=1)
    Sxy = F.conv2d(Ixy, kernel, padding=1)

    k_harris = 0.04
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    R = det - k_harris * (trace ** 2)

    R = R * mask
    R_flat = R.view(B, -1)
    _, indices = torch.topk(R_flat, k, dim=1)

    coords = torch.stack([indices // W, indices % W], dim=-1).float()
    coords = (coords / torch.tensor([H, W], device=images.device) - 0.5) * 2
    coords = coords.view(B, k, 2)
    return coords
    

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
    def __init__(self, k, patch_size):
        super().__init__()
        self.k = k
        self.patch_size = patch_size

    def forward(self, x):
        coords = harris_corner_detection_coords(x, self.k, patch_size=self.patch_size)
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
        self.classifier = nn.Linear(64 * k, num_classes)

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
            patch_size=patch_size
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

