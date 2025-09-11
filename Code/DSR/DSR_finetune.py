import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def vectorize(x, multichannel=False):
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)

def make_folder(name):
    if not os.path.exists(name):
        print('Creating folder: {}'.format(name))
        os.makedirs(name)

def check_for_gpu(device):
    if device.type == "cuda":
        if torch.cuda.is_available():
            print("GPU is available, running on GPU.\n")
        else:
            print("GPU is NOT available, running instead on CPU.\n")
    else:
        if torch.cuda.is_available():
            print("Warning: You have a CUDA device, so you may consider using GPU for potential acceleration\n by setting device to 'cuda'.\n")
        else:
            print("Running on CPU.\n")

def make_dataloader(x, y=None, batch_size=32, shuffle=True, num_workers=0):
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def energy_loss(x_true, x_est, beta=1, verbose=True):
    EPS = 0 if float(beta).is_integer() else 1e-5

    # Vectorize the true data (x_true) with multichannel=True
    x_true = vectorize(x_true, multichannel=True)  # Shape: [batch_size, 3, 4096]

    # Handle x_est (either a list of tensors or a single concatenated tensor)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))  # Split into a list of tensors
    m = len(x_est)

    # Vectorize each tensor in x_est (same as x_true)
    x_est = [vectorize(x_est[i],multichannel=True) for i in range(m)]
    x_est = torch.cat(x_est, dim=0)

    x_true_unseq = x_true.unsqueeze(1) #transform x_true to [batch_size, 1, 3, 4096]
    x_est_unseq  = x_est.unsqueeze(0) #transform x_est to [1, batch_size*sample_size, 3, 4096]

    # Calculate the distance between each estimated sample and the true sample (s1)
    s1 = (vector_norm(x_est_unseq - x_true_unseq, 2, dim=(2,3)) + EPS).pow(beta).mean()

    # Calculate the pairwise distance between estimated samples (s2)
    s2 = (torch.cdist(x_est_unseq, x_est_unseq, p=2) + EPS).pow(beta).mean() * m / (m - 1)

    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)


def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=True, weights=None):
    EPS = 0 if float(beta).is_integer() else 1e-5

    # Vectorize the inputs, flattening the spatial dimensions (D, H, W)
    x0 = vectorize(x0, multichannel=True)  # Shape: [batch_size, 3, 4096]
    x = vectorize(x, multichannel=True)
    xp = vectorize(xp, multichannel=True)

    if weights is None:
        weights = 1 / x0.size(0)  # If no weights are provided, assume uniform weights.

    if x0p is None:
        # Loss without x0p (case for standard two-sample comparison)
        s1 = ((vector_norm(x - x0, 2, dim=(1,2)) + EPS).pow(beta) * weights).sum() / 2 + \
             ((vector_norm(xp - x0, 2, dim=(1,2)) + EPS).pow(beta) * weights).sum() / 2
        s2 = ((vector_norm(x - xp, 2, dim=(1,2)) + EPS).pow(beta) * weights).sum()
        loss = s1 - s2 / 2
    else:
        # Loss with x0p (extended case with four terms)
        x0p = vectorize(x0p, multichannel=True)
        s1 = ((vector_norm(x - x0, 2, dim=(1,2)) + EPS).pow(beta).sum() +
              (vector_norm(xp - x0, 2, dim=(1,2)) + EPS).pow(beta).sum() +
              (vector_norm(x - x0p, 2, dim=(1,2)) + EPS).pow(beta).sum() +
              (vector_norm(xp - x0p, 2, dim=(1,2)) + EPS).pow(beta).sum()) / 4

        s2 = (vector_norm(x - xp, 2, dim=(1,2)) + EPS).pow(beta).sum()
        s3 = (vector_norm(x0 - x0p, 2, dim=(1,2)) + EPS).pow(beta).sum()

        loss = s1 - s2 / 2 - s3 / 2

    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss

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
        noisy = torch.randn_like(x) * (0.01 ** 0.5)
        x = x + noisy
        
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        # Apply both spatial and channel attention
        x = self.spatial_att(x) * self.se(x)
        return self.out(x)

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
        # Noisy
        #noise = torch.randn_like(x) * (0.01 ** 0.5)  # Noise with the same shape as x
        #x_noisy = x + noise  # Adding noise to the data
        
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dsr_fine_tune_step1(x, y, 
               lr=0.0001, num_epochs=500, batch_size=None,
               print_every_nepoch=1, print_times_per_epoch=1,
               device=device, standardize=True, verbose=True,
               load_path=None, freeze_layers=None):  # New params for fine-tuning
    if x.shape[0] != y.shape[0]:
        raise Exception("Sample size mismatch between covariates and response")
    
    dsr = DSR_fine_tune_step1(num_epochs=num_epochs, batch_size=batch_size,
                         standardize=standardize, device=device,
                         check_device=verbose, verbose=verbose, lr=lr,
                         load_path=load_path, freeze_layers=freeze_layers)
    
    dsr.train(x, y,num_epochs=num_epochs,
                   batch_size=batch_size, print_every_nepoch=print_every_nepoch,
                   print_times_per_epoch=print_times_per_epoch, verbose=verbose)
    return dsr

def dsr_fine_tune_step2(x, y, 
               lr=0.0001, num_epochs=500, batch_size=None,
               print_every_nepoch=1, print_times_per_epoch=1,
               device=device, standardize=True, verbose=True,
               load_path=None, freeze_layers=None):  # New params for fine-tuning
    if x.shape[0] != y.shape[0]:
        raise Exception("Sample size mismatch between covariates and response")
    
    dsr = DSR_fine_tune_step2(num_epochs=num_epochs, batch_size=batch_size,
                         standardize=standardize, device=device,
                         check_device=verbose, verbose=verbose, lr=lr,
                         load_path=load_path, freeze_layers=freeze_layers)
    
    dsr.train(x, y,num_epochs=num_epochs,
                   batch_size=batch_size, print_every_nepoch=print_every_nepoch,
                   print_times_per_epoch=print_times_per_epoch, verbose=verbose)
    return dsr

class DSR_fine_tune_step1(object):
    def __init__(self, num_epochs=20, batch_size=None, standardize=False,
                device=device, check_device=True, verbose=True,
                lr=0.0001, load_path=None, freeze_layers=None):  # Added fine-tuning params
        super().__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.standardize = standardize
        self.verbose = verbose
        self.lr = lr
        
        backbone = UNet3D(in_channels=3, out_channels=3, init_features=32).to(device)
        if load_path:
            ckpt = torch.load(load_path, map_location=device)
            state = ckpt.get('state_dict', ckpt)
            backbone.load_state_dict(state, strict=False)

        # 2) Freeze all pretrained weights
        for param in backbone.parameters():
            param.requires_grad = False

        backbone.conv = nn.Identity()
        new_head = AdvancedHead(feat_ch=32, mid_ch=128, out_ch=3, dropout_p=0.1).to(device)
        self.model = nn.Sequential(backbone, new_head)

        # 4) Build your optimizer over just the head’s parameters
        self.optimizer = optim.Adam(new_head.parameters(), lr=1e-4, weight_decay=1e-5)         

        # Tracking variables
        self.tr_loss = None
        self.best_val_loss = float('inf')
        self.early_stop_patience = 10
        self.no_improvement_count = 0

    def load_model(self, load_path):
        """Load pretrained weights while handling architecture mismatches"""
        try:
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if self.verbose:
                print(f"Loaded pretrained weights from {load_path}")
        except RuntimeError as e:
            print(f"Partial weights loaded: {str(e)}")
        except FileNotFoundError:
            print(f"Weights file {load_path} not found, initializing from scratch")

    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()
        
    def summary(self):
        """Print the model architecture and hyperparameters."""
        print("Engression model with\n" +
              "\t number of layers: {}\n".format(self.num_layer) +
              "\t hidden dimensions: {}\n".format(self.hidden_dim) +
              "\t number of epochs: {}\n".format(self.num_epochs) +
              "\t batch size: {}\n".format(self.batch_size) +
              "\t learning rate: {}\n".format(self.lr) +
              "\t standardization: {}\n".format(self.standardize) +
              "\t training mode: {}\n".format(self.model.training) +
              "\t device: {}\n".format(self.device))
        print("Training loss (original scale):\n" +
              "\t energy-loss: {:.2f}, \n\tE(|Y-Yhat|): {:.2f}, \n\tE(|Yhat-Yhat'|): {:.2f}".format(
                  self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
        
        
    def train(self, x, y, num_epochs=None, batch_size=None, lr=None, print_every_nepoch=1, print_times_per_epoch=1,verbose=False):
        
        self.train_mode()
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is None:
            batch_size = self.batch_size if self.batch_size is not None else x.size(0)
        if lr is not None:
            if lr != self.lr:
                self.lr = lr
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
        x = x.to(self.device)
        y = y.to(self.device)
        
        if batch_size >= x.size(0)//2:
            if verbose:
                print("Batch is larger than half of the sample size. Training based on full-batch gradient descent.")
            self.batch_size = x.size(0)
            for epoch_idx in range(self.num_epochs):
                self.model.zero_grad()
                y_sample1 = self.model(x)
                y_sample2 = self.model(x)
                #print(torch.allclose(y_sample1, y_sample2))
                loss, loss1, loss2 = energy_loss_two_sample(y, y_sample1, y_sample2, beta=1, verbose=True)
                loss.backward()
                self.optimizer.step()
                if (epoch_idx == 0 or  (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                    print("[Epoch {} ({:.0f}%)] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                        epoch_idx + 1, 100 * epoch_idx / num_epochs, loss.item(), loss1.item(), loss2.item()))
        else:
            train_loader = make_dataloader(x, y, batch_size=batch_size, shuffle=True)
            if verbose:
                print("Training based on mini-batch gradient descent with a batch size of {}.".format(batch_size))
            for epoch_idx in range(self.num_epochs):
                self.zero_loss()
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    self.train_one_iter(x_batch, y_batch)
                    if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                        if (batch_idx + 1) % ((len(train_loader) - 1) // print_times_per_epoch) == 0:
                            self.print_loss(epoch_idx, batch_idx)
    
    def zero_loss(self):
        self.tr_loss = 0
        self.tr_loss1 = 0
        self.tr_loss2 = 0
    
    def train_one_iter(self, x_batch, y_batch):
        self.model.zero_grad()
        y_sample1 = self.model(x_batch)
        y_sample2 = self.model(x_batch)
        #print(torch.allclose(y_sample1, y_sample2))
        loss, loss1, loss2 = energy_loss_two_sample(y_batch, y_sample1, y_sample2, beta=1, verbose=True)
        loss.backward()
        self.optimizer.step()
        self.tr_loss += loss.item()
        self.tr_loss1 += loss1.item()
        self.tr_loss2 += loss2.item()

        #y_samples = self.model.sample(val_x, sample_size=2, expand_dim=False)
        #val_loss = energy_loss(val_y, y_samples, beta=beta, verbose=True)
        
        #return val_loss
       
    def print_loss(self, epoch_idx, batch_idx, return_loss=False):
        loss_str = "[Epoch {} ({:.0f}%), batch {}] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
            epoch_idx + 1, 100 * epoch_idx / self.num_epochs, batch_idx + 1, 
            self.tr_loss / (batch_idx + 1), self.tr_loss1 / (batch_idx + 1), self.tr_loss2 / (batch_idx + 1))
        if return_loss:
            return loss_str
        else:
            print(loss_str)
    
    @torch.no_grad()
    def predict(self, x, target="mean", sample_size=5):
        
        self.eval_mode()  
        
        x = x.to(self.device)
        
        y_pred = self.model.predict(x, target, sample_size)
        
        return y_pred
    
    @torch.no_grad()
    def sample(self, x, sample_size=2, expand_dim=False):
        
        self.eval_mode()
        
        x = x.to(self.device)
        
        y_samples = self.model.sample(x, sample_size, expand_dim=expand_dim)            
        
        if sample_size == 1:
            y_samples = y_samples.squeeze(len(y_samples.shape) - 1)
        return y_samples
    
    
    @torch.no_grad()
    def eval_loss(self, x, y, loss_type="l2", sample_size=None, beta=1, verbose=False):
        
        if sample_size is None:
            sample_size = 2 if loss_type == "energy" else 100
        self.eval_mode()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "cor":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = cor(y, y_pred)
        elif loss_type == "l1":
            y_pred = self.predict(x, target=0.5, sample_size=sample_size)
            loss = (y - y_pred).abs().mean()
        else:
            assert loss_type == "energy"
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            loss = energy_loss(y, y_samples, beta=beta, verbose=verbose)
        if not verbose:
            return loss.item()
        else:
            loss, loss1, loss2 = loss
            return loss.item(), loss1.item(), loss2.item()

    def save_model(self, save_path="unet3d_model.pth"):
        """Enhanced model saving with architecture details"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "standardize": self.standardize,
            #"freeze_encoder": self.freeze_encoder,
            "val_loss": self.best_val_loss
        }
        torch.save(checkpoint, save_path)
        if self.verbose:
            print(f"Model saved with validation loss {self.best_val_loss:.4f}")


class DSR_fine_tune_step2(object):
    def __init__(self, num_epochs=20, batch_size=None, standardize=False,
                device='cpu', check_device=True, verbose=True,
                lr=0.0001, load_path=None, freeze_layers=None):  # Added fine-tuning params
        super().__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.standardize = standardize
        self.verbose = verbose
        self.lr = lr
        
        backbone = UNet3D(in_channels=3, out_channels=3, init_features=32).to(device)
        backbone.conv = nn.Identity()                          
        new_head = AdvancedHead(feat_ch=32, mid_ch=128, out_ch=3,
                                dropout_p=0.1).to(device)
        self.model = nn.Sequential(backbone, new_head)

        checkpoint = torch.load(load_path, map_location=device)
        state = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state, strict=False)

        # 3) Unfreeze everything for full fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True

        # 4) Build your optimizer over just the head’s parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)         

        # Tracking variables
        self.tr_loss = None
        self.best_val_loss = float('inf')
        self.early_stop_patience = 10
        self.no_improvement_count = 0

    def load_model(self, load_path):
        """Load pretrained weights while handling architecture mismatches"""
        try:
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if self.verbose:
                print(f"Loaded pretrained weights from {load_path}")
        except RuntimeError as e:
            print(f"Partial weights loaded: {str(e)}")
        except FileNotFoundError:
            print(f"Weights file {load_path} not found, initializing from scratch")

    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()
        
    def summary(self):
        """Print the model architecture and hyperparameters."""
        print("Engression model with\n" +
              "\t number of layers: {}\n".format(self.num_layer) +
              "\t hidden dimensions: {}\n".format(self.hidden_dim) +
              "\t number of epochs: {}\n".format(self.num_epochs) +
              "\t batch size: {}\n".format(self.batch_size) +
              "\t learning rate: {}\n".format(self.lr) +
              "\t standardization: {}\n".format(self.standardize) +
              "\t training mode: {}\n".format(self.model.training) +
              "\t device: {}\n".format(self.device))
        print("Training loss (original scale):\n" +
              "\t energy-loss: {:.2f}, \n\tE(|Y-Yhat|): {:.2f}, \n\tE(|Yhat-Yhat'|): {:.2f}".format(
                  self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
        
        
    def train(self, x, y, num_epochs=None, batch_size=None, lr=None, print_every_nepoch=1, print_times_per_epoch=1,verbose=False):
       
        self.train_mode()
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is None:
            batch_size = self.batch_size if self.batch_size is not None else x.size(0)
        if lr is not None:
            if lr != self.lr:
                self.lr = lr
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
        x = x.to(self.device)
        y = y.to(self.device)
        
        if batch_size >= x.size(0)//2:
            if verbose:
                print("Batch is larger than half of the sample size. Training based on full-batch gradient descent.")
            self.batch_size = x.size(0)
            for epoch_idx in range(self.num_epochs):
                self.model.zero_grad()
                y_sample1 = self.model(x)
                y_sample2 = self.model(x)
                #print(torch.allclose(y_sample1, y_sample2))
                loss, loss1, loss2 = energy_loss_two_sample(y, y_sample1, y_sample2, beta=1, verbose=True)
                loss.backward()
                self.optimizer.step()
                if (epoch_idx == 0 or  (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                    print("[Epoch {} ({:.0f}%)] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                        epoch_idx + 1, 100 * epoch_idx / num_epochs, loss.item(), loss1.item(), loss2.item()))
        else:
            train_loader = make_dataloader(x, y, batch_size=batch_size, shuffle=True)
            if verbose:
                print("Training based on mini-batch gradient descent with a batch size of {}.".format(batch_size))
            for epoch_idx in range(self.num_epochs):
                self.zero_loss()
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    self.train_one_iter(x_batch, y_batch)
                    if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                        if (batch_idx + 1) % ((len(train_loader) - 1) // print_times_per_epoch) == 0:
                            self.print_loss(epoch_idx, batch_idx)
    
    def zero_loss(self):
        self.tr_loss = 0
        self.tr_loss1 = 0
        self.tr_loss2 = 0
    
    def train_one_iter(self, x_batch, y_batch):
        self.model.zero_grad()
        y_sample1 = self.model(x_batch)
        y_sample2 = self.model(x_batch)
        #print(torch.allclose(y_sample1, y_sample2))
        loss, loss1, loss2 = energy_loss_two_sample(y_batch, y_sample1, y_sample2, beta=1, verbose=True)
        loss.backward()
        self.optimizer.step()
        self.tr_loss += loss.item()
        self.tr_loss1 += loss1.item()
        self.tr_loss2 += loss2.item()

        #y_samples = self.model.sample(val_x, sample_size=2, expand_dim=False)
        #val_loss = energy_loss(val_y, y_samples, beta=beta, verbose=True)
        
        #return val_loss
       
    def print_loss(self, epoch_idx, batch_idx, return_loss=False):
        loss_str = "[Epoch {} ({:.0f}%), batch {}] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
            epoch_idx + 1, 100 * epoch_idx / self.num_epochs, batch_idx + 1, 
            self.tr_loss / (batch_idx + 1), self.tr_loss1 / (batch_idx + 1), self.tr_loss2 / (batch_idx + 1))
        if return_loss:
            return loss_str
        else:
            print(loss_str)
    
    @torch.no_grad()
    def predict(self, x, target="mean", sample_size=5):
        
        self.eval_mode()  
        
        x = x.to(self.device)
        
        y_pred = self.model.predict(x, target, sample_size)
        
        return y_pred
    
    @torch.no_grad()
    def sample(self, x, sample_size=2, expand_dim=False):
        
        self.eval_mode()
        
        x = x.to(self.device)
        
        y_samples = self.model.sample(x, sample_size, expand_dim=expand_dim)            
        
        if sample_size == 1:
            y_samples = y_samples.squeeze(len(y_samples.shape) - 1)
        return y_samples
    
    
    @torch.no_grad()
    def eval_loss(self, x, y, loss_type="l2", sample_size=None, beta=1, verbose=False):
        
        if sample_size is None:
            sample_size = 2 if loss_type == "energy" else 100
        self.eval_mode()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "cor":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = cor(y, y_pred)
        elif loss_type == "l1":
            y_pred = self.predict(x, target=0.5, sample_size=sample_size)
            loss = (y - y_pred).abs().mean()
        else:
            assert loss_type == "energy"
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            loss = energy_loss(y, y_samples, beta=beta, verbose=verbose)
        if not verbose:
            return loss.item()
        else:
            loss, loss1, loss2 = loss
            return loss.item(), loss1.item(), loss2.item()

    def save_model(self, save_path="unet3d_model.pth"):
        """Enhanced model saving with architecture details"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "standardize": self.standardize,
            #"freeze_encoder": self.freeze_encoder,
            "val_loss": self.best_val_loss
        }
        torch.save(checkpoint, save_path)
        if self.verbose:
            print(f"Model saved with validation loss {self.best_val_loss:.4f}")