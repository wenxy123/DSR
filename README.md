# DSR
DSR is a distributional super-resolution method proposed in the paper "Distributional Deep Learning for Super-Resolution of 4D Flow MRI under Domain Shift." This repository provides the software implementation of DSR.

Our super-resolution approach is applied to regular cubic patches. Let the low-resolution cubic patch be denoted by $\mathbf{X} \in \mathbb{R}^{C \times D \times H \times W}$ and the corresponding high-resolution patch by $\mathbf{Y} \in \mathbb{R}^{C \times D \times H \times W}$, where $C$ is the number of channels (e.g., $C=3$ for 3D velocity components), and $D, H, W$ specify the spatial dimensions of the cubic patch.

We model the relationship between the high and low-resolution data using a multivariate pre-additive noise DSR model, which takes the following form:

$$\mathbf{Y} = \mathbf{h} \left( \mathbf{X} + \mathbf{\epsilon} \right),$$

where $\mathbf{h}$ denotes an unknown mapping function and $\mathbf{\epsilon}$ represents the noise term that independent of $\mathbf{X}$. 


## Usage Example

### Python
Below is a simple demonstration:
- Simulations: For details on DSR simulations, see [Simulation 1](https://github.com/wenxy123/DSR/blob/434bf55e47aee0d5da0ec251af437ff649f931cc/Code/Simulation/Simulation1/Simulation1.ipynb) and [Simulation 2](https://github.com/wenxy123/DSR/blob/434bf55e47aee0d5da0ec251af437ff649f931cc/Code/Simulation/Simulation2/Simulation2.ipynb), which correspond to the dense and sparse settings, respectively.
- Real data: For implementation on real data, start with [this tutorial](https://github.com/wenxy123/DSR/blob/434bf55e47aee0d5da0ec251af437ff649f931cc/Code/DSR/DSR_pretrain_data_preparation.ipynb) for preparing training data when the input is stored as 3D measurement points with spatial coordinates. Then, see [this tutorial](https://github.com/wenxy123/DSR/blob/434bf55e47aee0d5da0ec251af437ff649f931cc/Code/DSR/DSR_pretrain.ipynb) for model pretraining and [this tutorial](https://github.com/wenxy123/DSR/blob/434bf55e47aee0d5da0ec251af437ff649f931cc/Code/DSR/DSR_finetune.ipynb) for fine-tuning.

