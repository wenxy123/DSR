import torch
import numpy as np
import torch.nn as nn
import os
from torch.linalg import vector_norm
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm
import torch.optim as optim


## utility functions

def vectorize(x, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def cor(x, y):
    """Compute the correlation between two signals.

    Args:
        x (torch.Tensor): input data
        y (torch.Tensor): input data

    Returns:
        torch.Tensor: correlation between x and y
    """
    x = vectorize(x)
    y = vectorize(y)
    x = x - x.mean(0)
    y = y - y.mean(0)
    return ((x * y).mean()) / (x.std(unbiased=False) * y.std(unbiased=False))

def make_folder(name):
    """Make a folder.

    Args:
        name (str): folder name.
    """
    if not os.path.exists(name):
        print('Creating folder: {}'.format(name))
        os.makedirs(name)

def check_for_gpu(device):
    """Check if a CUDA device is available.

    Args:
        device (torch.device): current set device.
    """
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


def make_dataloader(x, y=None, batch_size=128, shuffle=True, num_workers=0):
    """Make dataloader.

    Args:
        x (torch.Tensor): data of predictors.
        y (torch.Tensor): data of responses.
        batch_size (int, optional): batch size. Defaults to 128.
        shuffle (bool, optional): whether to shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.

    Returns:
        DataLoader: data loader
    """
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def partition_data(x_full, y_full, cut_quantile=0.3, split_train="smaller"):
    """Partition data into training and test sets.

    Args:
        x_full (torch.Tensor): full data of x.
        y_full (torch.Tensor): full data of y.
        cut_quantile (float, optional): quantile of the cutting point of x. Defaults to 0.3.
        split_train (str, optional): which subset is used for for training. choices=["smaller", "larger"]. Defaults to "smaller".

    Returns:
        tuple of torch.Tensors: training and test data.
    """
    # Split data into training and test sets.
    x_cut = torch.quantile(x_full, cut_quantile)
    train_idx = x_full <= x_cut if split_train == "smaller" else x_full >= x_cut
    x_tr = x_full[train_idx]
    y_tr = y_full[train_idx]
    x_te = x_full[~train_idx]
    y_te = y_full[~train_idx]
    
    # Standardize data based on training statistics.
    x_tr_mean = x_tr.mean()
    x_tr_std = x_tr.std()
    y_tr_mean = y_tr.mean()
    y_tr_std = y_tr.std()
    x_tr = (x_tr - x_tr_mean)/x_tr_std
    y_tr = (y_tr - y_tr_mean)/y_tr_std
    x_te = (x_te - x_tr_mean)/x_tr_std
    y_te = (y_te - y_tr_mean)/y_tr_std
    x_full_normal = (x_full - x_tr_mean)/x_tr_std
    return x_tr.unsqueeze(1), y_tr.unsqueeze(1), x_te.unsqueeze(1), y_te.unsqueeze(1), x_full_normal         
            


def get_act_func(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid() 
    elif name == "tanh":
        return nn.Tanh() 
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "elu":
        return nn.ELU(inplace=True)
    else:
        return None

class InnerProductLayer(nn.Module):
    """A layer that computes an inner product with the input, where the first weight is fixed at 1, and the rest are learnable.

    Args:
        in_dim (int): input dimension
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        if in_dim < 1:
            raise ValueError("Input dimension must be at least 1")
        ## fix the first term at 1
        self.fixed_weight = torch.tensor(1.0)
        # The rest of the weights are learnable 
        if in_dim > 1:
            initial_values = torch.tensor([1.2, 1.5])
            self.learnable_weights = nn.Parameter(initial_values)
        else:
            self.learnable_weights = None

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        fixed_weight = self.fixed_weight.to(device=device, dtype=dtype)
        x_fixed = x[:, 0] * fixed_weight  
        if self.in_dim > 1:
            x_learnable = torch.matmul(x[:, 1:], self.learnable_weights)  
            x_reduced = x_fixed + x_learnable
        else:
            x_reduced = x_fixed  
        x_reduced = x_reduced.unsqueeze(1) 
        return x_reduced
    
    def get_weight_vector(self):
        """Returns the full weight vector (including fixed and learnable weights)."""
        device = self.fixed_weight.device
        dtype = self.fixed_weight.dtype
        fixed_weight = self.fixed_weight.to(device=device, dtype=dtype).detach().cpu().numpy()

        if self.learnable_weights is not None:
            learnable_weights = self.learnable_weights.detach().cpu().numpy()
            weight_vector = np.concatenate(([fixed_weight], learnable_weights))
        else:
            weight_vector = np.array([fixed_weight])

        return weight_vector

class Net(nn.Module):
    """Deterministic neural network.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=3, hidden_dim=100, 
                 add_bn=True, sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid
        
        net = [nn.Linear(in_dim, hidden_dim)]
        if add_bn:
            net += [nn.BatchNorm1d(hidden_dim)]
        net += [nn.ReLU(inplace=True)]
        for _ in range(num_layer - 2):
            net += [nn.Linear(hidden_dim, hidden_dim)]
            if add_bn:
                net += [nn.BatchNorm1d(hidden_dim)]
            net += [nn.ReLU(inplace=True)]
        net.append(nn.Linear(hidden_dim, out_dim))
        if sigmoid:
            out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
            net.append(out_act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MyNet(nn.Module):
    """Full neural network.

    Args:
        in_dim (int, optional): input dimension. 
        out_dim (int, optional): output dimension. 
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=3, hidden_dim=100, 
                 add_bn=True, sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid   
        
        self.inner_product = InnerProductLayer(in_dim=self.in_dim)
        self.net_layer = Net(in_dim=1, out_dim=out_dim, hidden_dim=hidden_dim, add_bn=add_bn, sigmoid=sigmoid)
    
    def forward(self,x):
        x = self.inner_product(x)
        x = self.net_layer(x)    
        return x
         
    
    def get_weight_vector(self):
        if hasattr(self, 'inner_product'):
            return self.inner_product.get_weight_vector()
        else:
            return None
        


def train_model(model, X_train, y_train, num_epochs=100, batch_size=32, learning_rate=1e-3, device=torch.device('cpu')):
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(dataloader.dataset)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')

def predict_model(model, X_test, device=torch.device('cpu')):
    model.eval()
    X_test = X_test.to(device)
    with torch.no_grad():
        outputs = model(X_test)
    return outputs.cpu()

def predict_g(model, X_test, device=torch.device('cpu')):
    model.eval()
    X_test = X_test.to(device)
    
    # Temporarily set the learnable weights to (1.2, 1.5)
    fixed_learnable_weights = torch.tensor([1.2, 1.5], dtype=torch.float32, device=device)
    
    with torch.no_grad():
       
        original_weights = model.inner_product.learnable_weights.clone().detach()
        model.inner_product.learnable_weights.data = fixed_learnable_weights
        outputs = model(X_test)
        model.inner_product.learnable_weights.data = original_weights

    return outputs.cpu()


