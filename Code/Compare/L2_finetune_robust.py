#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from scipy.ndimage import zoom  # For resampling
import math
#from pathos.multiprocessing import ProcessingPool as Pool

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.linalg import vector_norm


# In[5]:


# Custom collate function
def custom_collate_fn(batch):
    X_inputs, X_targets= zip(*batch)
    # Determine the maximum spatial dimensions in the batch
    max_C = max(x_input.shape[0] for x_input in X_inputs)
    max_D = max(x_input.shape[1] for x_input in X_inputs)
    max_H = max(x_input.shape[2] for x_input in X_inputs)
    max_W = max(x_input.shape[3] for x_input in X_inputs)

    # Pad all tensors to the maximum size
    X_inputs_padded = []
    X_targets_padded = []
    for x_input, x_target in zip(X_inputs, X_targets):
        padding_input = (
            0, max_W - x_input.shape[3],  # Width padding
            0, max_H - x_input.shape[2],  # Height padding
            0, max_D - x_input.shape[1],  # Depth padding
        )
        padding_target = (
            0, max_W - x_target.shape[3],
            0, max_H - x_target.shape[2],
            0, max_D - x_target.shape[1],
        )
        x_input_padded = F.pad(x_input, padding_input, mode='constant', value=0)
        x_target_padded = F.pad(x_target, padding_target, mode='constant', value=0)
        X_inputs_padded.append(x_input_padded)
        X_targets_padded.append(x_target_padded)

    X_inputs_batch = torch.stack(X_inputs_padded)
    X_targets_batch = torch.stack(X_targets_padded)
    #ts_batch = torch.tensor(ts)
    return X_inputs_batch, X_targets_batch


# In[7]:


class MyDataset(Dataset):
    """Custom dataset for training inputs and targets."""
    def __init__(self, train_input, train_target):
        """
        Args:
            train_input (torch.Tensor): Training data input of shape (12000, 3, 16, 16, 16).
            train_target (torch.Tensor): Training data target of shape (12000, 3, 16, 16, 16).
        """
        self.train_input = train_input
        self.train_target = train_target

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.train_input)

    def __getitem__(self, idx):
        # Return the input-target pair at the given index
        input_tensor = self.train_input[idx]
        target_tensor = self.train_target[idx]
        return input_tensor, target_tensor


# In[9]:


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=32,dropout_p=0.2):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, dropout_p)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.encoder2 = UNet3D._block(features, features * 2, dropout_p)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, dropout_p)


        self.bottleneck = UNet3D._block(features * 4, features * 8, dropout_p)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 8, features * 4, dropout_p)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 4, features * 2, dropout_p)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, dropout_p)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, dropout_p=0.2):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)  # [N, features, D, H, W]
        enc2 = self.encoder2(self.pool1(enc1))  # [N, features*2, D/2, H/2, W/2]
        enc3 = self.encoder3(self.pool2(enc2))  # [N, features*4, D/4, H/4, W/4]

        bottleneck = self.bottleneck(enc3)  # [N, features*8, D/4, H/4, W/4]

        dec3 = self.upconv3(bottleneck)  # [N, features*4, D/2, H/2, W/2]
        # Adjust dec3 size if necessary
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='trilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)  # [N, features*2, D, H, W]
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)  # [N, features, 2D, 2H, 2W]
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1) # dim = [32,32,16,16,16]

        return self.conv(dec1)


class ResBlock3D(nn.Module):
    """A basic 3D residual block."""
    def __init__(self, channels):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)

class SpatialAttention3D(nn.Module):
    """Spatial attention mechanism for 3D feature maps."""
    def __init__(self):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att

class AdvancedHead(nn.Module):
    """
    A deeper and more expressive head:
      - Projects feature channels to mid_ch
      - Two residual blocks
      - Spatial + channel (SE) attention
      - Final 1x1 conv to out_ch
    """
    def __init__(self, feat_ch=32, mid_ch=128, out_ch=3, dropout_p=0.1):
        super(AdvancedHead, self).__init__()
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv3d(feat_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )
        # Two residual blocks
        self.res1 = ResBlock3D(mid_ch)
        self.res2 = ResBlock3D(mid_ch)
        # Spatial attention
        self.spatial_att = SpatialAttention3D()
        # Squeeze-and-excitation (channel attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(mid_ch, mid_ch // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch // 16, mid_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Final output projection
        self.out = nn.Conv3d(mid_ch, out_ch, kernel_size=1)

    def forward(self, x):
        #noisy = torch.randn_like(x) * (0.01 ** 0.5)
        #x = x + noisy
        
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        # Apply both spatial and channel attention
        x = self.spatial_att(x) * self.se(x)
        return self.out(x)
    
# In[11]:


def L2_fine_tune_step1(model, train_dataloader, device, criterion, freeze_layers=None, lr=1e-4, num_epochs=10, verbose=True):
    
    # 1. Move model to device
    model = model.to(device)
    
    new_head = AdvancedHead(feat_ch=32, mid_ch=128, out_ch=3, dropout_p=0.1).to(device)
    optimizer = optim.Adam(new_head.parameters(), lr=1e-4, weight_decay=1e-5)

    # 4. Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]")

        with tqdm(total=len(train_dataloader), desc=f"Fine-tuning Epoch {epoch+1}", unit="batch", disable=not verbose) as pbar:
            for X_input, X_target in train_dataloader:
                X_input = X_input.to(device)
                X_target = X_target.to(device)

                # Forward pass
                outputs = model(X_input)

                # Ensure outputs and X_target have the same shape
                # (Useful if upsampling or downsampling is required)
                if outputs.shape != X_target.shape:
                    X_target = F.interpolate(X_target, size=outputs.shape[2:], mode='trilinear', align_corners=False)

                # Compute loss
                loss = criterion(outputs, X_target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)

        avg_loss = epoch_loss / len(train_dataloader)
        if verbose:
            print(f"Average Loss: {avg_loss:.4f}")

    return model

def L2_fine_tune_step2(model, train_dataloader, device, criterion, freeze_layers=None, lr=1e-4, num_epochs=10, verbose=True):
    
    # 1. Move model to device
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)   

    # 4. Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]")

        with tqdm(total=len(train_dataloader), desc=f"Fine-tuning Epoch {epoch+1}", unit="batch", disable=not verbose) as pbar:
            for X_input, X_target in train_dataloader:
                X_input = X_input.to(device)
                X_target = X_target.to(device)

                # Forward pass
                outputs = model(X_input)

                # Ensure outputs and X_target have the same shape
                # (Useful if upsampling or downsampling is required)
                if outputs.shape != X_target.shape:
                    X_target = F.interpolate(X_target, size=outputs.shape[2:], mode='trilinear', align_corners=False)

                # Compute loss
                loss = criterion(outputs, X_target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)

        avg_loss = epoch_loss / len(train_dataloader)
        if verbose:
            print(f"Average Loss: {avg_loss:.4f}")

    return model


# In[15]:


# Load the model
def load_model(model, model_path, device):
    # Load the model's state dict from the saved file
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


# In[17]:


import scipy.io

# Load the .mat file
vcdf_mat = scipy.io.loadmat("/data/VCFD.mat")

Velocity_mat = scipy.io.loadmat("/data/Velocity.mat")

vcdf_data = vcdf_mat['Velocity_CFD'][0]
vel_data = Velocity_mat['Velocity'][0]

vcdf_data_flat = vcdf_data.reshape(-1, 3)

# Create a boolean mask that is True for rows that are not all zeros.
mask1 = ~np.all(vcdf_data_flat == 0, axis=1)
mask0 = np.all(vcdf_data_flat == 0, axis=1)
vcdf_data1 = vcdf_data_flat[mask1]

vel_data_flat = vel_data.reshape(-1, 3)
vel_data1 = vel_data_flat[mask1]


# In[19]:


x = 46
y = 46
z = 168

vcdf_data2 = vcdf_data1.reshape(x, y, z, 3)
vel_data2 = vel_data1.reshape(x, y, z, 3)

data_input = torch.Tensor(vel_data2)
data_target = torch.Tensor(vcdf_data2)

pad_tuple = (0, 0,  # +0 on dim #3 -> remains 3
             4, 4,  # +8 on dim #2 -> 168 + 8 = 176
             1, 1,  # +2 on dim #1 -> 46 + 2 = 48
             1, 1)  # +2 on dim #0 -> 46 + 2 = 48

# Apply padding
data_input_expanded = F.pad(data_input, pad_tuple)
data_target_expanded = F.pad(data_target, pad_tuple)

# Step 2: Generate cubic patches of shape [16, 16, 16, 3] from both data1 and data2
patch_size = 16

# Prepare lists to store the patches for both data1 (X_inputs) and data2 (X_targets)
X_inputs_patches = []
X_targets_patches = []

# Loop through the expanded data to extract cubic patches
for i in range(0, data_input_expanded.shape[0], patch_size):  # Loop along the first dimension (80)
    for j in range(0, data_input_expanded.shape[1], patch_size):  # Loop along the second dimension (912)
        for k in range(0, data_input_expanded.shape[2], patch_size):  # Loop along the third dimension (240)
            # Extract patches from data1 (X_inputs) and data2 (X_targets)
            patch_data1 = data_input_expanded[i:i+patch_size, j:j+patch_size, k:k+patch_size, :]
            patch_data2 = data_target_expanded[i:i+patch_size, j:j+patch_size, k:k+patch_size, :]
            
            if patch_data1.shape == torch.Size([patch_size, patch_size, patch_size, 3]):  # Ensure the patch is correct
                X_inputs_patches.append(patch_data1)
                X_targets_patches.append(patch_data2)

# Convert the list of patches into tensors
X_inputs_tensor = torch.stack(X_inputs_patches)
X_targets_tensor = torch.stack(X_targets_patches)


# In[21]:


X_inputs_tensor1 = torch.permute(X_inputs_tensor, (0, 4, 1, 2, 3))
X_targets_tensor1 = torch.permute(X_targets_tensor, (0, 4, 1, 2, 3))
X_inputs_tensor1.shape


# In[23]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N_sim = 30

MSE_X = []
MSE_Y = []
MSE_Z = []
MSE_all = []


for i in range(N_sim):
    np.random.seed(i)

    n_samples = 15
       
    # Generate random indices
    indices = np.random.choice(X_inputs_tensor1.shape[0], size=n_samples, replace=False)
    print(indices)
    
    # Subsample data
    train_x_input = X_inputs_tensor1[indices]
    train_x_target = X_targets_tensor1[indices]

    sigma = 0.01
    epsilon = torch.randn_like(train_x_input) * (sigma ** 0.5)

    train_x_input = train_x_input + epsilon
    
    N = X_inputs_tensor1.shape[0]
    
    mask = np.ones(N, dtype=bool)
    
    mask[indices] = False
    
    test_x_input = X_inputs_tensor1[mask]
    test_x_target = X_targets_tensor1[mask]

    train_dataset = MyDataset(train_x_input, train_x_target)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    model = UNet3D(in_channels=3, out_channels=3)
    model = load_model(model,'/data/L2_unet_model_1000_noise01_200.pth',device)
    criterion = nn.MSELoss()
    
    L2_fine_tuned_model_step1 = L2_fine_tune_step1(
            model=model,
            train_dataloader=train_dataloader,
            device=device,
            criterion=criterion,   
            lr=1e-4,
            num_epochs=300,
            verbose=True)
    
    model_path = 'L2_unet_model_1000_noise01_200_tune_robust_step1.pth'
    torch.save(L2_fine_tuned_model_step1.state_dict(), model_path)
    
    load_path = 'L2_unet_model_1000_noise01_200_tune_robust_step1.pth'
    backbone = UNet3D(in_channels=3, out_channels=3, init_features=32).to(device)
    backbone.conv = nn.Identity()                          
    new_head = AdvancedHead(feat_ch=32, mid_ch=128, out_ch=3,
                            dropout_p=0.1).to(device)
    model = nn.Sequential(backbone, new_head)

    checkpoint = torch.load(load_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=False)

    for param in model.parameters():
        param.requires_grad = True
        
    L2_fine_tuned_model_step2 = L2_fine_tune_step2(
        model=model,
        train_dataloader=train_dataloader,
        device=device,
        criterion=criterion,   
        lr=1e-4,
        num_epochs=200,
        verbose=True
    )
    
    model_path = 'L2_unet_model_1000_noise01_200_tune_robust_step2.pth'
    torch.save(L2_fine_tuned_model_step2.state_dict(), model_path)

    backbone = UNet3D(in_channels=3, out_channels=3, init_features=32).to(device)
    backbone.conv = nn.Identity()
    new_head = AdvancedHead(feat_ch=32, mid_ch=128, out_ch=3, dropout_p=0.1).to(device)
    model = nn.Sequential(backbone, new_head)
    model = load_model(model,'L2_unet_model_1000_noise01_200_tune_robust_step2.pth',device)
    model.to(device)
    model.eval()

    batch_size = 32

    #n_samples = X_inputs_tensor1.shape[0]
    n_samples = test_x_input.shape[0]
    
    # A placeholder for gathering predictions
    compare_predictions = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = test_x_input[start_idx:end_idx].to(device)
        epsilon_t = torch.randn_like(batch) * (sigma ** 0.5)
        input_batch = batch + epsilon_t
        
        compare_preds = model(input_batch)
        compare_predictions.append(compare_preds)
    
    compare_predictions = torch.cat(compare_predictions, dim=0)  # shape: [2535, 10]
    compare_predictions1 = compare_predictions.permute(0,2,3,4,1)
    test_x_input1 = test_x_input.permute(0,2,3,4,1)
    test_x_target1 = test_x_target.permute(0,2,3,4,1)
    compare_predictions_flat = compare_predictions1.reshape(-1, 3)
    test_flat = test_x_target1.reshape(-1, 3)
    
    MSE_X0 = mean_squared_error(compare_predictions_flat.detach().cpu().numpy()[:,0], test_flat.detach().numpy()[:,0])
    MSE_Y0 = mean_squared_error(compare_predictions_flat.detach().cpu().numpy()[:,1], test_flat.detach().numpy()[:,1])
    MSE_Z0 = mean_squared_error(compare_predictions_flat.detach().cpu().numpy()[:,2], test_flat.detach().numpy()[:,2])
    MSE_all0 = mean_squared_error(torch.norm(compare_predictions_flat,dim=1).detach().cpu().numpy(), torch.norm(test_flat,dim=1).detach().cpu().numpy())        

    MSE_X.append(MSE_X0)
    MSE_Y.append(MSE_Y0)
    MSE_Z.append(MSE_Z0)
    MSE_all.append(MSE_all0)


compare_mse_df = pd.DataFrame(
    data = {
        'MSE_X':MSE_X,
        'MSE_Y':MSE_Y,
        'MSE_Z':MSE_Z,
        'MSE_all':MSE_all,
    }
)

compare_mse_df['Type'] = 'L2'

compare_mse_df.to_csv('L2_mse.csv')



