#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:53:09 2024

@author: isekulic
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
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples_6000.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load surrogate model
F_surrogate = torch.load('/scratch/htc/btian/NNs_test/Forward_surrogate_model_35000_sobol.pt', map_location=device)

def scale_input(X):
    X_scaled = (X - lower) / (upper - lower) * 2.0 - 1.0  # Normalize inputs to be from -1 to 1
    return X_scaled

def mean_std(predictions):
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    return mean_pred, std_pred

# Calculate predictions on a testing set
Xtest_scaled = scale_input(Xtest).float().to(device)
print(Xtest_scaled.device)

predicted_diff_ord = F_surrogate(Xtest_scaled)  # Forward pass on scaled data gives scaled predictions

# Load ground truth data
ground_truth_test = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test_6000.pt', map_location=device)
ground_truth_train = torch.load('/scratch/htc/btian/NNs_test/ground_truth_train_35000_sobol.pt', map_location=device)

# Unscale predictions
mean_pred, std_pred = mean_std(ground_truth_train)
predicted_diff_ord = predicted_diff_ord * std_pred.repeat(1, No_NN_red) + mean_pred.repeat(1, No_NN_red)
predicted_diff_ord = predicted_diff_ord.reshape(Xtest.shape[0], No_NN_red, num_outputs).transpose(0, 1)
print(torch.mean(predicted_diff_ord, 0)[0:20])
print(ground_truth_test[0:20])

error = torch.sum(torch.abs(ground_truth_test - torch.mean(predicted_diff_ord, 0))) / (Xtest.shape[0] * num_outputs)
print('Average error on test data per diff order:', error.item())

# Load training data
Xtrain = torch.load('/scratch/htc/btian/NNs_test/train_samples.pt').float()

# Define the backward model

class BackwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BackwardModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.layer7 = nn.Linear(hidden_size, hidden_size)
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        x = self.output(x)
        return x




trainloss = []
testloss = []



# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create the backward model instance and move to device
input_size = 5  # Number of output neurons from the forward model
hidden_size = 512
output_size = len(lower)  # Dimension of the input space (number of input features for the forward model)

backward_model = BackwardModel(input_size, hidden_size, output_size).to(device)


model_save_path = '/scratch/htc/btian/NNs_test/trained_backward_model.pth'
backward_model.load_state_dict(torch.load(model_save_path))
print(f"Loaded backward model from {model_save_path}")

# Evaluation part
backward_model.eval()
with torch.no_grad():
    X_pred = backward_model(ground_truth_test).to(device)

# Get Y predictions from the forward model using Xtest
Y_test_scaled = F_surrogate(Xtest_scaled.float()).to(device)
Y_test_pred = Y_test_scaled * std_pred.repeat(1, No_NN_red) + mean_pred.repeat(1, No_NN_red)
Y_test_pred = Y_test_pred.reshape(Xtest.shape[0], No_NN_red, num_outputs).transpose(0, 1)

# Get Y predictions from the forward model using X_pred
Y_pred_scaled = F_surrogate(X_pred.float()).to(device)
Y_pred = Y_pred_scaled * std_pred.repeat(1, No_NN_red) + mean_pred.repeat(1, No_NN_red)
Y_pred = Y_pred.reshape(Xtest.shape[0], No_NN_red, num_outputs).transpose(0, 1)


ground_truth_test2 = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt', map_location=device)
print(ground_truth_test.shape)
print(ground_truth_test2.shape)




# Calculate errors
error_true_test = torch.sum(torch.abs(ground_truth_test - torch.mean(Y_test_pred, 0))) / (Xtest.shape[0] * num_outputs)
error_test_pred = torch.sum(torch.abs(torch.mean(Y_test_pred, 0) - torch.mean(Y_pred, 0))) / (Xtest.shape[0] * num_outputs)
error_true_pred = torch.sum(torch.abs(ground_truth_test - torch.mean(Y_pred, 0))) / (Xtest.shape[0] * num_outputs)

print('Average error on test data true-test (Y_test_pred):', error_true_test.item())
print('Average error on test data test-pred (Y_pred):', error_test_pred.item())
print('Average error on test data true-pred (Y_test_pred):', error_true_pred.item())

# Mean Absolute Error
loss = nn.L1Loss()
error_true_test = loss(torch.mean(Y_test_pred, 0), ground_truth_test)
error_test_pred = loss(torch.mean(Y_pred, 0), torch.mean(Y_test_pred, 0))
error_true_pred = loss(ground_truth_test, torch.mean(Y_pred, 0))

print('MAE on test data true-test (Y_test_pred):', error_true_test.item())
print('MAE on test data test-pred (Y_pred):', error_test_pred.item())
print('MAE on test data true-pred (Y_test_pred):', error_true_pred.item())

# Print the first 10 flux values
print("First 10 flux values:")
for i in range(10):
    print(f"Real flux (Y): {ground_truth_test[i]}")
    print(f"Y prediction on Xtest: {torch.mean(Y_test_pred, 0)[i]}")
    print(f"Y prediction on BackwardModel's Xpred: {torch.mean(Y_pred, 0)[i]}")
    print("---")



import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming ground_truth_test and predictions (Y_test_pred, Y_pred) are already loaded from your previous code
# And the following variables are defined:
# - ground_truth_test: shape [6000, 5]
# - Y_test_pred: predictions using Xtest
# - Y_pred: predictions using X_pred

# Mean Absolute Error for each vector
loss = nn.L1Loss(reduction='none')
mae_true_test = loss(torch.mean(Y_test_pred, 0), ground_truth_test).mean(dim=1)
mae_test_pred = loss(torch.mean(Y_pred, 0), torch.mean(Y_test_pred, 0)).mean(dim=1)
mae_true_pred = loss(ground_truth_test, torch.mean(Y_pred, 0)).mean(dim=1)

# Convert to numpy for plotting
mae_true_test_np = mae_true_test.cpu().detach().numpy()
mae_test_pred_np = mae_test_pred.cpu().detach().numpy()
mae_true_pred_np = mae_true_pred.cpu().detach().numpy()
print(mae_true_test_np)
# Plot the distribution
plt.figure(figsize=(14, 7))
sns.violinplot(data=[mae_true_test_np, mae_test_pred_np, mae_true_pred_np], cut=0)
plt.xticks([0, 1, 2], ['True-Test', 'Test-Pred', 'True-Pred'])
plt.xlabel('Error Type')
plt.ylabel('Mean Absolute Error')
plt.yscale('log')
plt.title('Distribution of Mean Absolute Errors')
plt.savefig('/scratch/htc/btian/NNs_test/log MAE distribution.png')