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
num_epochs = 10000

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


G.load_state_dict(torch.load('/scratch/htc/btian/NNs_test/trained_generator.pth', map_location=device))
G.eval()  # Set the generator to evaluation mode

# Load your test data
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt').to(device)
Ytest = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt', map_location=device)

# Generate predictions for the test set
with torch.no_grad():  # Disable gradient calculation for evaluation
    noise = torch.randn(Ytest.size(0), noise_dim).to(device)  # Generate random noise
    X_pred = G(Ytest.float(), noise)  # Generate radius predictions

# Compare the predictions with the actual ground truth
print("First 10 Test Results:")
for i in range(10):
    print(f"Test sample {i+1}:")
    print(f"Real R (Ytest): {Xtest[i]}")
    print(f"Predicted R (Y_pred): {X_pred[i]}")
    print("---")

# Calculate and print Mean Absolute Error (MAE) as a simple performance metric
mae_loss = nn.L1Loss()
mae = mae_loss(X_pred, Xtest.to(device))
print(f"Mean Absolute Error on test set: {mae.item()}")







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 18:45:54 2024

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



flux = compute_flux(Xtest, F_surrogate, lower_bound, upper_bound)

pflux = compute_flux(X_pred, F_surrogate, lower_bound, upper_bound)








# Assuming compute_flux returns NumPy arrays for flux and pflux
flux_np = compute_flux(Xtest, F_surrogate, lower_bound, upper_bound)  # NumPy array
pflux_np = compute_flux(X_pred, F_surrogate, lower_bound, upper_bound)  # NumPy array

# Convert the NumPy arrays to PyTorch tensors
flux = torch.from_numpy(flux_np).float().to(device)  # Convert to float tensor and move to the appropriate device
pflux = torch.from_numpy(pflux_np).float().to(device)  # Convert to float tensor and move to the appropriate device

# Now flux and pflux are PyTorch tensors
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have already computed flux and pflux as torch tensors

# Calculate MAE across the 5 dimensions for each of the 2000 test samples
mae_per_sample = nn.L1Loss(reduction='none')(flux, pflux).mean(dim=1)

# Convert MAE to NumPy array for plotting
mae_per_sample_np = mae_per_sample.cpu().detach().numpy()
print("First 10 Test Results:")
for i in range(10):
    print(f"Test sample {i+1}:")
    print(f"Real R flux (Ytest): {flux[i]}")
    print(f"Predicted R flux (Y_pred): {pflux[i]}")
    print(f"loss: {mae_per_sample_np[i]}")
    print("---")

import torch





# Calculate and print Mean Absolute Error (MAE) as a simple performance metric
mae_loss = nn.L1Loss()
mae = mae_loss(flux, pflux)
print(f"Mean Absolute Error on test set: {mae.item()}")



# Calculate the mean of the actual flux values
flux_mean = torch.mean(flux, dim=0)

# Calculate the total sum of squares (SST)
sst = torch.sum((flux - flux_mean) ** 2)

# Calculate the residual sum of squares (SSR)
ssr = torch.sum((flux - pflux) ** 2)

# Calculate R^2
r2 = 1 - ssr / sst
r2_value = r2.mean().item()  # Take the mean if you want an average R^2 for all dimensions

print(f"R^2 on the test set: {r2_value}")



# Plot the distribution of the MAE for each test sample
plt.figure(figsize=(10, 6))
sns.violinplot(data=mae_per_sample_np, inner="quartile", scale="width")
plt.yscale("log")  # Use log scale for better visibility across different magnitudes
plt.title("Distribution of Mean Absolute Errors")
plt.xlabel("Test Samples")
plt.ylabel("Mean Absolute Error")

# Save the plot to a file
output_path = "../../../../Downloads/btian/OneDmetalense_opt_example/mae_distribution.png"  # Example path
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Optionally, display the plot
plt.show()

plt.savefig('/scratch/htc/btian/NNs_test/CGAN log MAE distribution.png')






import torch
import numpy as np

# Load your test data
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt').to(device)  # True radius
Ytest = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt', map_location=device)  # True flux

# Select only the first 5 flux samples from Ytest
selected_samples = Ytest[:5]

# Define the range for R values
lower_bound = np.array([180.0, 180.0, 180.0, 140.0, 200.0, 180.0])
upper_bound = np.array([240.0, 240.0, 240.0, 200.0, 260.0, 240.0])

# Number of different R values to generate for each flux
num_variations = 5

for i in range(5):  # Iterate over the first 5 flux samples
    print(f"Test sample {i+1} (Flux):")
    
    # Print the true radius and flux
    print(f"  True Radius (R): {Xtest[i].cpu().numpy()}")
    print(f"  True Flux (Y): {Ytest[i].cpu().numpy()}")
    
    R_list = []
    pflux_list = []

    for j in range(num_variations):
        with torch.no_grad():  # Disable gradient calculation for evaluation
            noise = torch.randn(1, noise_dim).to(device)  # Generate random noise
            R_pred = G(selected_samples[i].unsqueeze(0).float(), noise.float())  # Generate one R prediction
            R_list.append(R_pred.squeeze(0).cpu().numpy())

            # Calculate the predicted flux from this generated R
            R_pred_np = R_pred.cpu().numpy()
            flux_pred = compute_flux(R_pred_np, F_surrogate, lower_bound, upper_bound)
            pflux_list.append(flux_pred)

            print(f"  Variation {j+1}:")
            print(f"  Predicted R (Radius): {R_list[-1]}")
            print(f"  Predicted Flux (Y) by this R: {pflux_list[-1]}")
            print("---")

    print("---" * 10)

































import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# Define the constant input flux
input_flux = torch.tensor([0.03, 0.04, 0.05, 0.06, 0.07], dtype=torch.float).unsqueeze(0).to(device)

# Number of variations
num_variations = 100

# Arrays to store predicted fluxes, corresponding radii, and MAE
predicted_fluxes = []
predicted_radii = []
mae_errors = []

# MAE loss function
mae_loss = nn.L1Loss()

print("Generating 100 flux variations and calculating MAE:")

for j in range(num_variations):
    with torch.no_grad():
        noise = torch.randn(1, noise_dim).to(device)  # Generate random noise
        R_pred = G(input_flux.float(), noise.float())  # Generate one R prediction

        # Store the predicted radius
        predicted_radii.append(R_pred.squeeze(0).cpu().numpy())

        # Calculate the predicted flux from this generated R
        R_pred_np = R_pred.cpu().numpy()
        flux_pred = compute_flux(R_pred_np, F_surrogate, lower_bound, upper_bound)
        predicted_fluxes.append(flux_pred)

        # Calculate the MAE between the predicted flux and the true flux
        flux_tensor = torch.tensor(flux_pred, dtype=torch.float).to(device)
        true_flux_tensor = torch.tensor(input_flux.cpu().numpy(), dtype=torch.float).to(device)
        mae_error = mae_loss(flux_tensor, true_flux_tensor)
        mae_errors.append(mae_error.item())

        print(f"Variation {j+1}: MAE = {mae_error.item()}")

# Convert lists to numpy arrays for easier processing
predicted_fluxes_np = np.array(predicted_fluxes)
predicted_radii_np = np.array(predicted_radii)
mae_errors_np = np.array(mae_errors)

# Define a tolerance for considering radii to be "too close"
radius_tolerance = 3  # Adjust this value based on your specific requirements

# Function to check if a radius is sufficiently different from previously selected radii
def is_sufficiently_different(radius, selected_radii, tolerance):
    return all(np.linalg.norm(radius - selected) > tolerance for selected in selected_radii)

# List to store indices of the best 3 variations with different radii
best_indices = []
selected_radii = []

# Find the best 3 variations with sufficiently different radii
for idx in mae_errors_np.argsort():
    if len(best_indices) < 3:
        radius = predicted_radii_np[idx]
        if is_sufficiently_different(radius, selected_radii, radius_tolerance):
            best_indices.append(idx)
            selected_radii.append(radius)

print(f"\nBest 3 variations (with smallest MAE and different radii) are at indices: {best_indices}")

# Print the corresponding radii for the best 3 variations
for i, idx in enumerate(best_indices):
    print(f"Variation {i+1}:")
    print(f"  MAE: {mae_errors_np[idx]:.4f}")
    print(f"  Corresponding Radius: {predicted_radii_np[idx]}")
    print(f"  Predicted Flux: {predicted_fluxes_np[idx].flatten()}")
    print("---")

# Plot histogram of the true flux and best 3 variations
plt.figure(figsize=(10, 6))
diffraction_orders = np.arange(1, 6)  # Assuming 5 diffraction orders

# Plotting the true flux
plt.bar(diffraction_orders - 0.3, input_flux.cpu().numpy().flatten(), width=0.2, label="True Flux", color='blue')

# Plotting the best 3 variations
colors = ['orange', 'green', 'red']
for i, idx in enumerate(best_indices):
    plt.bar(diffraction_orders + 0.2*i, predicted_fluxes_np[idx].flatten(), width=0.2, label=f'Variation {i+1} (MAE: {mae_errors_np[idx]:.4f})', color=colors[i])

plt.xlabel('Diffraction Orders')
plt.ylabel('Flux')
plt.title('Comparison of True Flux and Best 3 Predicted Flux Variations with Different Radii')
plt.legend()
plt.grid(True)

# Set Y-axis limits
plt.ylim(0, 0.1)

# Save the plot to a file
output_path = "/scratch/htc/btian/NNs_test/best_flux_comparison_histogram_different_radii.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Comparison histogram saved to: {output_path}")





