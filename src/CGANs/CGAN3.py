#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:08:23 2024

@author: btian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:41:35 2024

@author: btian
"""
import sys
sys.path.append("/scratch/htc/btian/OneD_DotP/")
sys.path.append("/scratch/htc/mmccarver/aotoolbox")
import os
import numpy as np
import torch
import DotProjector_geom_surrogate
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim



def scale_input(X):
    X_scaled = (X - lower) / (upper - lower) * 2.0 - 1.0  # Normalize inputs to be from -1 to 1
    return X_scaled

def mean_std(predictions):
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    return mean_pred, std_pred

# Set environment variable
os.environ['JCMROOT'] = '/scratch/htc/bzfhamme/JCMsuite/JCMsuite.6.2.1'

# Initialize surrogate model
supercell = DotProjector_geom_surrogate.Fixed_Heights()

# Configuration
No_NN_red = 60  # Number of nets in reduced ensemble
num_outputs = 5  # Number of output neurons in each ensemble member

# Domain values
lower = np.asarray([info["domain"][0] for info in supercell.domain])
upper = np.asarray([info["domain"][1] for info in supercell.domain])

# Load testing data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load surrogate model
F_surrogate = torch.load('/scratch/htc/btian/NNs_test/Forward_surrogate_model_35000_sobol.pt', map_location=device)

# Load ground truth data
ground_truth_test = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test_6000.pt', map_location=device)
ground_truth_train = torch.load('/scratch/htc/btian/NNs_test/ground_truth_train_35000_sobol.pt', map_location=device)

# Unscale predictions
mean_pred, std_pred = mean_std(ground_truth_train)

import numpy as np
import torch

def compute_flux(R_array, model, lower_bound, upper_bound, num_nets=60, num_outputs=5):
    """
    计算给定输入数组的flux。

    参数:
    - R_array: 输入的numpy数组，形状为(n, 6)
    - model: 训练好的模型
    - lower_bound: 输入参数的下边界，numpy数组
    - upper_bound: 输入参数的上边界，numpy数组
    - num_nets: 神经网络模型中使用的网络数量 (默认: 60)
    - num_outputs: 每个模型的输出神经元数量 (默认: 5)
    
    返回:
    - flux: 输出的flux值，numpy数组，形状为(n, 5)
    """

    # 将输入的numpy数组转换为torch张量
    Xtest = torch.tensor(R_array, dtype=torch.double).cpu()

    # 输入归一化
    Xtest_scaled = (Xtest - lower_bound) / (upper_bound - lower_bound) * 2.0 - 1.0
    
    # 前向传播计算
    predicted_diff_ord = model(Xtest_scaled.float().to(device))

    # 从模型中获取ground truth的均值和标准差
    mean_pred, std_pred = mean_std(ground_truth_train)
    
    # 反归一化
    predicted_diff_ord = predicted_diff_ord * std_pred.repeat(1, num_nets) + mean_pred.repeat(1, num_nets)
    predicted_diff_ord = predicted_diff_ord.reshape(Xtest.shape[0], num_nets, num_outputs).transpose(0, 1)
    
    # 计算flux的均值
    flux = torch.mean(predicted_diff_ord, 0)
    
    # 将结果转换回numpy数组
    flux_numpy = flux.cpu().detach().numpy()

    return flux_numpy

# 定义R的范围
lower_bound = np.array([180.0, 180.0, 180.0, 140.0, 200.0, 180.0])
upper_bound = np.array([240.0, 240.0, 240.0, 200.0, 260.0, 240.0])













#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:41:35 2024

@author: btian
"""

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define your Generator and Discriminator as before
class Generator(nn.Module):
    def __init__(self, input_dim, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
        
    def forward(self, x, noise):
        x = torch.cat([x, noise], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + output_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)

# Hyperparameters
noise_dim = 100
input_dim = 5
output_dim = 6
lr = 0.0002
batch_size = 64
num_epochs = 20000

# Load your data
ground_truth_train = torch.load('/scratch/htc/btian/NNs_test/ground_truth_train.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
Xtrain = torch.load('/scratch/htc/btian/NNs_test/train_samples.pt').to(device)

# Create TensorDataset and DataLoader
dataset = TensorDataset(Xtrain, ground_truth_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the models, loss function, and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(input_dim, noise_dim, output_dim).to(device)
D = Discriminator(input_dim, output_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)
















mae_values = []


Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt').to(device)
Ytest = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt', map_location=device)
# Training loop
for epoch in range(num_epochs):
    for i, (R, flux) in enumerate(dataloader):
        batch_size = flux.size(0)
        
        # Real labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Compute loss with real samples
        outputs = D(flux, R)
        d_loss_real = criterion(outputs, real_labels)
        
        # Generate fake samples
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_R = G(flux, noise)
        outputs = D(flux, fake_R.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        # Generate fake samples and compute loss
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_R = G(flux, noise)
        outputs = D(flux, fake_R)
        g_loss = criterion(outputs, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
    if epoch % 100 == 0:

        G.eval()  # Set the Generator to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for evaluation
            # Load your test data

            
            noise = torch.randn(Ytest.size(0), noise_dim).to(device)  # Generate random noise
            X_pred = G(Ytest, noise)  # Generate radius predictions
            
            # Compute flux from predicted radius and actual radius
            flux_pred = compute_flux(X_pred.cpu().numpy(), F_surrogate, lower_bound, upper_bound)
            flux_actual = compute_flux(Xtest.cpu().numpy(), F_surrogate, lower_bound, upper_bound)

            # Convert the NumPy arrays to PyTorch tensors
            flux_pred = torch.from_numpy(flux_pred).float().to(device)
            flux_actual = torch.from_numpy(flux_actual).float().to(device)

            # Calculate and print Mean Absolute Error (MAE)
            mae_loss = nn.L1Loss()
            mae = mae_loss(flux_pred, flux_actual)
            mae_values.append(mae.item())  # Store the MAE value
            # Calculate R^2
            flux_mean = torch.mean(flux_actual, dim=0)
            sst = torch.sum((flux_actual - flux_mean) ** 2)
            ssr = torch.sum((flux_actual - flux_pred) ** 2)
            r2 = 1 - ssr / sst
            r2_value = r2.mean().item()  # Average R^2 over all dimensions

            print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, MAE: {mae.item():.4f}, R^2: {r2_value:.4f}')

        G.train()  # Set the Generator back to training mode

# Save the trained Generator and Discriminator models
torch.save(G.state_dict(), '/scratch/htc/btian/NNs_test/trained_generator4.pth')
torch.save(D.state_dict(), '/scratch/htc/btian/NNs_test/trained_discriminator4.pth')
print("Models saved to /scratch/htc/btian/NNs_test/")





# Plot MAE vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(range(0, num_epochs, 100), mae_values)
plt.title('Test MAE Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.grid(True)
plt.savefig('/scratch/htc/btian/NNs_test/CGAN_mae_loss_vs_epoch.png')