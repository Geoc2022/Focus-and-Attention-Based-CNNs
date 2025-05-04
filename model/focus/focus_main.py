import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import focus_main

image_size = 28
patch_size = 28
k_points = 3

epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
# test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

model = focus_main.KeypointPatchModel(k=k_points, patch_size=patch_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


for epoch in range(epochs):
    train_loss, train_acc = focus_main.train_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

test_loss, test_acc = focus_main.evaluate(model, test_loader, criterion)
print("\n")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")