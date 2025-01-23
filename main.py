import os
import random
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
print(torch.__version__)

# ----------------------------
#        CONFIG
# ----------------------------
class Config:
    # Paths
    train_image_dir = 'C:/Users/39327/Desktop/Research/Unet-transformer/dataset/data/train/images'   # e.g. 'train/images'
    train_mask_dir  = 'C:/Users/39327/Desktop/Research/Unet-transformer/dataset/data/train/labels'    # e.g. 'train/masks'
    val_image_dir   = 'C:/Users/39327/Desktop/Research/Unet-transformer/dataset/data/val/images'     # e.g. 'val/images'
    val_mask_dir    = 'C:/Users/39327/Desktop/Research/Unet-transformer/dataset/data/val/labels'       # e.g. 'val/masks'

    # Data
    img_size = 256
    batch_size = 1

    # Training
    lr = 1e-4
    num_epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
#    Dataset & Transform
# ----------------------------
class PolypDataset(Dataset):
    """
    A simple dataset for polyp segmentation.
    Expects directory structure:
      - image_dir: containing .jpg or .png images
      - mask_dir:  containing corresponding .jpg or .png segmentation masks
    Filenames should match up so that:
      image: images/xxx.png
      mask:  masks/xxx.png
    """
    def __init__(self, image_dir, mask_dir, img_size=256, transform=None):
        super().__init__()
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png'))) + sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.mask_paths  = sorted(glob(os.path.join(mask_dir, '*.png'))) + sorted(glob(os.path.join(mask_dir, '*.jpg')))

        assert len(self.image_paths) == len(self.mask_paths), "Image/Mask count mismatch!"
        
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')   # grayscale

        # Resize
        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = transforms.ToTensor()(image)

        # Convert mask to tensor (0 or 1)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()  # binarize

        return image, mask

# We can define a basic augmentation or just normalization transform
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        # Example normalization if your dataset is in [0..255]
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

# ----------------------------
#    Transformer Blocks
# ----------------------------
class TransformerBlock(nn.Module):
    """
    A small Transformer block with Multi-Head Self-Attention (MHSA) + Feed Forward.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, N, C) -> batch, sequence_len, embed_dim
        # MHSA
        h1 = self.norm1(x)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + self.drop(attn_out)
        
        # FF
        h2 = self.norm2(x)
        ff_out = self.ff(h2)
        x = x + self.drop(ff_out)
        return x

def rearrange_spatial_to_batch(x):
    """
    Flatten spatial dimensions into sequence dimension for attention.
    Suppose x has shape (B, C, H, W).
    We want to transform it to (B, H*W, C) so that each pixel is a "token."
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H*W)             # shape: (B, C, H*W)
    x = x.permute(0, 2, 1)           # shape: (B, H*W, C)
    return x

def rearrange_batch_to_spatial(x, H, W):
    """
    Inverse of rearrange_spatial_to_batch.
    x shape: (B, H*W, C)
    return shape: (B, C, H, W)
    """
    B, N, C = x.shape
    x = x.permute(0, 2, 1)           # shape: (B, C, H*W)
    x = x.view(B, C, H, W)           # shape: (B, C, H, W)
    return x

# ----------------------------
#    UNet Components
# ----------------------------
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscale with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ----------------------------
#    Transformer UNet
# ----------------------------
class TransformerUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, img_size=256, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        A U-Net style architecture with a transformer block in the bottleneck.
        """
        super(TransformerUNet, self).__init__()
        self.img_size = img_size

        # Encoder
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck: We apply a Transformer block on the 512 feature maps
        self.transformer_embed_dim = 512  # must match the channel dimension from down3
        self.transformer_block = TransformerBlock(embed_dim=self.transformer_embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout)
        
        # Optional: Another down or just direct transform
        self.down4 = Down(512, 512)

        # Decoder
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # (B, 64,  H,   W)
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        x4 = self.down3(x3)  # (B, 512, H/8, W/8)
        
        # Bottleneck Transformer
        bnx = self.down4(x4) # shape: (B, 512, H/16, W/16)
        
        # Flatten spatial dims for transformer
        B, C, H, W = bnx.shape
        t = rearrange_spatial_to_batch(bnx)   # (B, H*W, 512)
        t = self.transformer_block(t)         # (B, H*W, 512)
        bnx = rearrange_batch_to_spatial(t, H, W) # (B, 512, H, W)

        # Decoder
        x = self.up1(bnx, x4)  # 512 -> 256
        x = self.up2(x, x3)    # 256 -> 128
        x = self.up3(x, x2)    # 128 -> 64
        x = self.up4(x, x1)    # 64  -> 64

        logits = self.outc(x)  # (B, out_ch, H, W)
        return logits

# ----------------------------
#    Loss Functions
# ----------------------------
def dice_loss(pred, target, smooth=1e-5):
    """
    Dice loss (soft) for binary segmentation
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def bce_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

def combined_loss(pred, target, alpha=0.5):
    """
    Combine BCE + Dice
    """
    return alpha*dice_loss(pred, target) + (1-alpha)*bce_loss(pred, target)

# ----------------------------
#     Training & Validation
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def validate_one_epoch(model, loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(loader)

# ----------------------------
#         Main
# ----------------------------
def main():
    cfg = Config()

    # Datasets
    train_transform = get_transform()
    val_transform   = get_transform()
    
    train_dataset = PolypDataset(
        cfg.train_image_dir, 
        cfg.train_mask_dir, 
        img_size=cfg.img_size, 
        transform=train_transform
    )
    val_dataset = PolypDataset(
        cfg.val_image_dir, 
        cfg.val_mask_dir, 
        img_size=cfg.img_size, 
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Model
    model = TransformerUNet(
        in_ch=3, 
        out_ch=1, 
        img_size=cfg.img_size,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1
    ).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_loss = validate_one_epoch(model, val_loader, cfg.device)

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

if __name__ == "__main__":
    main()
