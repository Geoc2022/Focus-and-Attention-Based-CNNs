import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

def visualize_attention(I_train, a, up_factor, no_attention=False):
    img = I_train.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=8, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    img = cv2.resize(img, (attn.shape[1], attn.shape[0]))
    if no_attention:
        return torch.from_numpy(img)
    else:
        vis = 0.6 * img + 0.4 * attn
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
    def __init__(
        self,
        in_features_l,
        in_features_g,
        attn_features,
        up_factor,
        normalize_attn=True,
    ):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        
        # Corrected channel dimensions
        self.W_l = nn.Conv2d(
            in_channels=in_features_l,
            out_channels=attn_features,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.W_g = nn.Conv2d(
            in_channels=in_features_g,
            out_channels=attn_features,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.phi = nn.Conv2d(
            in_channels=attn_features,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        
        if self.up_factor > 1:
            g_ = F.interpolate(
                g_, scale_factor=self.up_factor, mode="bilinear", align_corners=False
            )
        
        c = self.phi(F.relu(l_ + g_))  # batch_sizex1xWxH

        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
            
        f = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)
        else:
            output = F.adaptive_avg_pool2d(f, (1, 1)).view(N, C)
        return a, output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
            normalize_attn=True,
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256*4*4 + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1000) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        s1 = x  #save for attention
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        s2 = x  #save for attention
        x = self.pool(x)  
        
        a, g = self.attn(s1, s2)
        
        x = torch.flatten(x, 1) 
        x = torch.cat((x, g), dim=1)
        
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

    viz_batch = next(iter(test_loader))
    viz_images, _ = viz_batch[0][:4].to(device), viz_batch[1][:4].to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            # test_loss += F.nll_loss(
            #     output, target, reduction="sum"
            # ).item()  # sum up batch loss
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(
                dim=1, keepdim=True
            ) 
            correct += pred.eq(target.view_as(pred)).sum().item()
        _, attn_maps = model(viz_images)
        I_train = utils.make_grid(viz_images.cpu(), nrow=4, normalize=True)
        
        orig = visualize_attention(I_train, attn_maps[:,:4], up_factor=2, no_attention=True) 
        first = visualize_attention(I_train, attn_maps[:,:4], up_factor=2, no_attention=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.imshow(orig)
        ax2.imshow(first)
        ax1.set_title('Input Images (First 4)')
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
    return test_loss, correct / len(test_loader.dataset)
