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


os.environ['JCMROOT']= '/scratch/htc/bzfhamme/JCMsuite/JCMsuite.6.2.1'


supercell = DotProjector_geom_surrogate.Fixed_Heights()

# Here we load the trained model-surrogate and do some predictions

No_NN_red = 60 #number of nets in reduced ensemble
num_outputs = 5 #number of output neurons in each ensemble member

#lower domain values
lower = np.asarray(([info["domain"][0] for info in supercell.domain]))
#upper domain values
upper = np.asarray(([info["domain"][1] for info in supercell.domain]))

#loading testing data
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


F_surrogate = torch.load('/scratch/htc/btian/NNs_test/Forward_surrogate_model_35000_sobol.pt', map_location=device)

def scale_input(X):
    X_scaled = torch.empty(X.size())

    X_scaled = (
     (X - lower) / (upper - lower)
    ) * 2.0 - 1.0  # normalizing the inputs to be from -1 to 1

    return X_scaled

def mean_std(predictions):
    
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    return mean_pred, std_pred

# calculate predictions on a testing set
Xtest_scaled = scale_input(Xtest).float().to(device)
print(Xtest_scaled.device)
predicted_diff_ord = F_surrogate(Xtest_scaled) # forward pass on scaled data gives scaled predictions

# %%
ground_truth_test = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt', map_location=device)

ground_truth_train = torch.load('/scratch/htc/btian/NNs_test/ground_truth_train_35000_sobol.pt', map_location=device)

mean_pred , std_pred = mean_std(ground_truth_train)
predicted_diff_ord = predicted_diff_ord * std_pred.repeat(1 , No_NN_red) + mean_pred.repeat(1 , No_NN_red) #unscaling
predicted_diff_ord = predicted_diff_ord.reshape(Xtest.shape[0], No_NN_red, num_outputs).transpose(0,1) 
print(torch.mean(predicted_diff_ord, 0)[0:20])
print(ground_truth_test[0:20])

error = torch.sum(torch.abs(ground_truth_test - torch.mean(predicted_diff_ord, 0))) / (Xtest.shape[0] * 5)

print('Average error on test data per diff order:', error.item())

Xtrain = torch.load('/scratch/htc/btian/NNs_test/train_samples.pt').float()
ground_truth_test = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt', map_location=device)
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt').float().to(device)


import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn

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
hidden_size = 300
output_size = len(lower)  # Dimension of the input space (number of input features for the forward model)

backward_model = BackwardModel(input_size, hidden_size, output_size).to(device)






# Loss function and optimizer
criterion = nn.L1Loss()  # Use L1 loss (Mean Absolute Error)
optimizer = optim.Adam(backward_model.parameters(), lr=0.001)


# Move data to device
ground_truth_train = ground_truth_train.to(device)
mean_pred = mean_pred.to(device)
std_pred = std_pred.to(device)

# Training loop for the tandem model
num_epochs = 10000


for epoch in range(num_epochs):
    backward_model.train()
    
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass: Backward model to get X_pred from Y
    X_pred = backward_model(ground_truth_train).to(device)

    # Forward pass: Pre-trained forward model to get Y_pred from X_pred
    Y_pred_scaled = F_surrogate(X_pred.float()).to(device)
    
    # Unscale the predictions
    Y_pred = Y_pred_scaled * std_pred.repeat(1, No_NN_red).to(device) + mean_pred.repeat(1, No_NN_red).to(device)
    Y_pred = Y_pred.reshape(35000, No_NN_red, num_outputs).transpose(0, 1)

    # Compute the loss (MSE between real Y and generated Y)
    loss = criterion(torch.mean(Y_pred, 0), ground_truth_train)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Calculate test loss
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        backward_model.eval()
        with torch.no_grad():
            X_pred_test = backward_model(ground_truth_test).to(device)
        
        Y_pred_test_scaled = F_surrogate(X_pred_test.float()).to(device)
        Y_pred_test = Y_pred_test_scaled * std_pred.repeat(1, No_NN_red).to(device) + mean_pred.repeat(1, No_NN_red).to(device)
        Y_pred_test = Y_pred_test.reshape(Xtest.shape[0], No_NN_red, num_outputs).transpose(0, 1)
        
        test_loss = criterion(torch.mean(Y_pred_test, 0), ground_truth_test)
        
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        trainloss.append(loss.item())
        testloss.append(test_loss.item())

print("Training complete.")















# Plot the training and test loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(trainloss, label='Train Loss')
plt.plot(testloss, label='Test Loss')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.savefig('/scratch/htc/btian/NNs_test/tandem_loss_curves.png')
print("Loss curves saved to /scratch/htc/btian/NNs_test/tandem_loss_curves.png")


# Save the trained model
model_save_path = '/scratch/htc/btian/NNs_test/trained_backward_model.pth'
torch.save(backward_model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

# %%
# Evaluation part

# Get predictions from the backward model using test data
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

# Calculate errors
error_true_test = torch.sum(torch.abs(ground_truth_test - torch.mean(Y_test_pred, 0))) / (Xtest.shape[0] * 5)
error_test_pred = torch.sum(torch.abs(torch.mean(Y_test_pred, 0)- torch.mean(Y_pred, 0))) / (Xtest.shape[0] * 5)
error_true_pred = torch.sum(torch.abs(ground_truth_test - torch.mean(Y_pred, 0))) / (Xtest.shape[0] * 5)

print('Average error on test data true-test (Y_test_pred):', error_true_test.item())
print('Average error on test data test-pred (Y_pred):', error_test_pred.item())
print('Average error on test data true-pred (Y_test_pred):', error_true_pred.item())







loss = nn.L1Loss()




error_true_test = loss(torch.mean(Y_test_pred, 0),ground_truth_test)
error_test_pred = loss(torch.mean(Y_pred, 0),torch.mean(Y_test_pred, 0))
error_true_pred = loss(ground_truth_test,torch.mean(Y_pred, 0))

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







