#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# In[2]:


import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from scipy.ndimage import zoom  # For resampling
import math 

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.linalg import vector_norm


# In[23]:


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


# In[10]:


# Define the 3D UNet model
class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=32, dropout_p=0.2):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features, dropout_p)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.encoder2 = self._block(features, features * 2, dropout_p)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.encoder3 = self._block(features * 2, features * 4, dropout_p)

        self.bottleneck = self._block(features * 4, features * 8, dropout_p)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4, dropout_p)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2, dropout_p)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, dropout_p)

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
        # ... forward logic remains the same ...
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(enc3)

        dec3 = self.upconv3(bottleneck)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='trilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


# In[18]:


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


# In[12]:


train_input = torch.load('/data/train_input_1000_noise01.pt')
train_input.shape


# In[13]:


train_target = torch.load('/data/train_target_1000_noise01.pt')


# In[16]:


valid_input = torch.load('/data/valid_input_1000_noise01.pt')
valid_input.shape


# In[17]:


valid_target = torch.load('/data/valid_target_1000_noise01.pt')


# In[24]:


# Create the custom dataset
train_dataset = MyDataset(train_input, train_target)


# In[25]:


train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate_fn)


# In[27]:


valid_dataset = MyDataset(valid_input, valid_target)
valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)


# In[ ]:


# Training function
def train(model, train_dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        with tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch+1}", unit="batch") as pbar:
            for X_input, X_target in train_dataloader:
                X_input = X_input.to(device)
                X_target = X_target.to(device)
                # Forward pass
                outputs = model(X_input)
                # Ensure outputs and X_target have the same shape
                if outputs.shape != X_target.shape:
                    X_target = F.interpolate(X_target, size=outputs.shape[2:], mode='trilinear', align_corners=False)
                loss = criterion(outputs, X_target)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Average Loss: {avg_loss:.4f}")


# In[28]:


def evaluate_loss_on_validation(model, valid_dataloader, criterion, device="cpu"):
    """
    Evaluates the loss on the validation dataset.
    
    Args:
        model: The trained model.
        valid_dataloader: DataLoader for the validation dataset.
        criterion: The loss function.
        device: The device to run the model on, either "cpu" or "cuda".
    
    Returns:
        avg_val_loss: The average loss on the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize total loss
    num_batches = len(valid_dataloader)  # Number of batches in the validation dataloader

    with torch.no_grad():  # No gradients are needed for validation
        for x_val, y_val in valid_dataloader:
            # Move data to the correct device (e.g., GPU or CPU)
            x_val, y_val = x_val.to(device), y_val.to(device)

            # Perform forward pass
            preds = model(x_val)

            # Calculate the loss for the batch
            loss = criterion(preds, y_val)
            val_loss += loss.item()  # Accumulate the loss for each batch

    avg_val_loss = val_loss / num_batches  # Average loss over all validation batches
    print(f"Validation Loss: {avg_val_loss:.4f}")  # Print the average loss
    
    return avg_val_loss


# In[ ]:


model = UNet3D(in_channels=3, out_channels=3)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 200  # Adjust the number of epochs as needed
train(model, train_dataloader, optimizer, criterion, device, num_epochs)


# Save the trained model
model_path = 'L2_unet_model_1000_noise01_200.pth'
#model_path = 'unet3d_super_resolution_norm.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to '{model_path}'")

