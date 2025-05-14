import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

def plot_attention(I_train, a, up_factor, no_attention=False):
    img = I_train.permute((1, 2, 0)).cpu().numpy()    
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2) 

    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=4, normalize=True, scale_each=True)
    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
    if no_attention:
        return torch.from_numpy(img)
    else:
        heatmap = plt.cm.jet(attn[..., 0])[..., :3]
        heatmap = np.float32(heatmap)
        
        img = np.clip(img, 0, 1)
        img_resized = np.zeros_like(heatmap)
        for c in range(3):
            img_resized[..., c] = np.clip(
                F.interpolate(
                    torch.tensor(img[..., c]).unsqueeze(0).unsqueeze(0),
                    size=(heatmap.shape[0], heatmap.shape[1]),
                    mode='bilinear'
                ).squeeze().numpy(),
                0, 1
            )
        
        vis = 0.6 * img_resized + 0.4 * heatmap
        return torch.from_numpy(vis)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        
        self.W_l = nn.Conv2d(in_features_l, attn_features, 1, bias=False)
        self.W_g = nn.Conv2d(in_features_g, attn_features, 1, bias=False)
        self.phi = nn.Conv2d(attn_features, 1, 1, bias=True)
        
    def forward(self, l, g):
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        
        c = self.phi(F.relu(l_ + g_)) 
        a = torch.sigmoid(c)           
        
        attended_features = a * l
        
        return a, attended_features


class Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.attn = AttentionBlock(
            in_features_l=128,
            in_features_g=256,
            attn_features=128,
            up_factor=2,
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(27392 if in_channels == 1 else 36864, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1000) 

    def forward(self, x):
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)  # Repeat grayscale channel 3 times
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        s1 = x
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        s2 = x 
        x = self.pool(x)
        
        a, g = self.attn(s1, s2)

        x_flat = torch.flatten(x, 1)
        g_flat = torch.flatten(g, 1)
        
        x = torch.cat((x_flat, g_flat), dim=1) 
        
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1), a

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    print(enumerate(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
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

    num_ims = 5
    im_batch = next(iter(test_loader))
    rand_indices = torch.randperm(im_batch[0].size(0))[:num_ims]
    out_ims = im_batch[0][rand_indices].to(device)

    with torch.no_grad():
        for data, target in test_loader:
            if data.size(1) == 1:
                data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        _, attn_maps = model(out_ims)        
        orig = []
        heatmaps = []
        for i in range(num_ims):
            single_attn = attn_maps[i].unsqueeze(0)
            single_img = out_ims[i].unsqueeze(0)
            single_grid = utils.make_grid(single_img.cpu(), nrow=1, normalize=True)
            orig.append(plot_attention(single_grid, single_attn, up_factor=2, no_attention=True))
            heatmaps.append(plot_attention(single_grid, single_attn, up_factor=2, no_attention=False))
        orig = torch.cat(orig, dim=1)
        heatmaps = torch.cat(heatmaps, dim=1)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.imshow(orig)
        ax2.imshow(heatmaps)
        ax1.set_title('Original images')
        ax2.set_title('Attention Maps')
        plt.tight_layout()
        plt.show()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )