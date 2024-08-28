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
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

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
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples_6000.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load surrogate model
F_surrogate = torch.load('/scratch/htc/btian/NNs_test/Forward_surrogate_model_35000_sobol.pt', map_location=device)

# Load ground truth data
ground_truth_test = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test_6000.pt', map_location=device)
ground_truth_train = torch.load('/scratch/htc/btian/NNs_test/ground_truth_train_35000_sobol.pt', map_location=device)

# Unscale predictions
mean_pred, std_pred = mean_std(ground_truth_train)

def compute_flux(R_array, model, lower_bound, upper_bound, num_nets=60, num_outputs=5):
    """
    Compute the flux for a given input array.

    Parameters:
    - R_array: Input numpy array with shape (n, 6)
    - model: Trained model
    - lower_bound: Lower bound of the input parameters, numpy array
    - upper_bound: Upper bound of the input parameters, numpy array
    - num_nets: Number of networks used in the neural network model (default: 60)
    - num_outputs: Number of output neurons in each model (default: 5)
    
    Returns:
    - flux: Output flux value, numpy array with shape (n, 5)
    """

    # Convert the input numpy array to a torch tensor
    Xtest = torch.tensor(R_array, dtype=torch.double)

    # Normalize the input
    Xtest_scaled = (Xtest - lower_bound) / (upper_bound - lower_bound) * 2.0 - 1.0
    
    # Forward pass
    predicted_diff_ord = model(Xtest_scaled.float().to(device))

    # Get mean and standard deviation of ground truth from the model
    mean_pred, std_pred = mean_std(ground_truth_train)
    
    # Unnormalize
    predicted_diff_ord = predicted_diff_ord * std_pred.repeat(1, num_nets) + mean_pred.repeat(1, num_nets)
    predicted_diff_ord = predicted_diff_ord.reshape(Xtest.shape[0], num_nets, num_outputs).transpose(0, 1)
    
    # Compute mean flux
    flux = torch.mean(predicted_diff_ord, 0)
    
    # Convert the result back to a numpy array
    flux_numpy = flux.cpu().detach().numpy()

    return flux_numpy

# Define the flux you want to achieve
input_flux = torch.tensor([0.07, 0.07, 0.07, 0.07, 0.07], dtype=torch.float).unsqueeze(0).to(device)

# CGAN model definition
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

# Load the pre-trained Generator model
G = Generator(input_dim=5, noise_dim=100, output_dim=6).to(device)
G.load_state_dict(torch.load('/scratch/htc/btian/NNs_test/trained_generator.pth', map_location=device))
G.eval()

# Generate initial radii using CGANs
num_variations = 1000
predicted_radii = []

for _ in range(num_variations):
    noise = torch.randn(1, 100).to(device)  # noise_dim = 100
    R_pred = G(input_flux, noise)
    predicted_radii.append(R_pred.squeeze(0).cpu().detach().numpy())

predicted_radii_np = np.array(predicted_radii)

# Define the range of R
lower_bound = np.array([180.0, 180.0, 180.0, 140.0, 200.0, 180.0])
upper_bound = np.array([240.0, 240.0, 240.0, 200.0, 260.0, 240.0])

# Define objective function to calculate the MAE between predicted flux and target flux
def objective(R):
    flux = compute_flux(R.reshape(1, -1), F_surrogate, lower_bound, upper_bound)
    mae_error = np.mean(np.abs(flux - input_flux.cpu().detach().numpy().flatten()))
    return mae_error

# Local minimum search function
def local_search(R_initial, lower_bound, upper_bound):
    res = minimize(objective, R_initial, method='Nelder-Mead', bounds=list(zip(lower_bound, upper_bound)))
    return res.x, res.fun

# Iterative search
valid_Rs = []
for i, R_initial in enumerate(predicted_radii_np):
    print(f"Iteration {i+1}: Starting local optimization from CGAN-generated radius.")
    R_local_min, mae_value = local_search(R_initial, lower_bound, upper_bound)
    if mae_value < 0.01:  # Adjust this threshold as necessary
        valid_Rs.append(R_local_min)

print(f"Number of valid R samples found: {len(valid_Rs)}")


# Define the search with increasing radius
def search_with_increasing_radius(R, initial_radius, max_radius, step, threshold):
    current_radius = initial_radius
    best_radius = 0  # Record the largest valid radius
    best_samples = []  # Record the samples at the best radius

    while current_radius <= max_radius:
        # Generate points within the current radius
        local_samples = R + np.random.uniform(-current_radius, current_radius, size=(100, 6))

        # Check if all points are valid
        all_valid = True
        for R_local in local_samples:
            mae_value = objective(R_local)
            if mae_value >= threshold:
                all_valid = False
                break

        # If all points are valid, update the maximum valid radius and continue increasing the radius
        if all_valid:
            best_radius = current_radius
            best_samples = local_samples  # Record the samples at the current radius
            current_radius += step
        else:
            # Once an invalid point is found, stop and return the current best radius and samples
            break

    return best_radius, best_samples

# Main search loop, recording the maximum valid radius for each R
radius_record = []
extended_Rs = []
all_best_samples = []  # Record the best samples for each R

initial_radius = 0.001  # Initial radius
max_radius = 5  # Maximum radius
step = 0.02  # Step size for increasing the radius
threshold = 0.002  # Threshold for determining validity

for i, R in enumerate(valid_Rs):
    print(f"Iteration {i+1}: Performing search with increasing radius.")

    # Perform the search with increasing radius for each R
    best_radius, best_samples = search_with_increasing_radius(R, initial_radius, max_radius, step, threshold)
    radius_record.append(best_radius)
    all_best_samples.append(best_samples)

    # Points generated within the best radius are considered valid, add them to extended_Rs
    if best_radius > 0:
        extended_Rs.extend(best_samples)

# Output the maximum valid radius for each R
for idx, radius in enumerate(radius_record):
    print(f"R[{idx}] maximum valid radius: {radius}")

# Find the R with the maximum radius
max_radius_index = np.argmax(radius_record)
max_radius_R = valid_Rs[max_radius_index]
max_radius_value = radius_record[max_radius_index]
max_radius_samples = np.array(all_best_samples[max_radius_index])

print(f"\nR with the largest valid radius is at index {max_radius_index}, coordinates: {max_radius_R}")
print(f"Corresponding maximum valid radius: {max_radius_value}")

# Compute the flux at the center position
center_flux = compute_flux(max_radius_R.reshape(1, -1), F_surrogate, lower_bound, upper_bound)
print(f"Flux at the center R: {center_flux}")

# Compute and print the flux for a few of the most distant positions within the valid radius
num_samples_to_print = min(3, len(max_radius_samples))  # Print up to 3 samples or fewer if fewer exist
distances = np.linalg.norm(max_radius_samples - max_radius_R, axis=1)
most_distant_indices = np.argsort(distances)[-num_samples_to_print:]  # Get indices of most distant samples

for i, idx in enumerate(most_distant_indices):
    distant_flux = compute_flux(max_radius_samples[idx].reshape(1, -1), F_surrogate, lower_bound, upper_bound)
    print(f"Flux at most distant position {i+1}: {distant_flux} (distance: {distances[idx]})")

# Define dimension pairs to plot
dimension_pairs = [(0, 1), (2, 3), (4, 5)]
dimension_names = [("Radius1", "Radius2"), ("Radius3", "Radius4"), ("Radius5", "Radius6")]

# Plot 2D scatter plots for each dimension pair
for idx, (dim_pair, dim_names) in enumerate(zip(dimension_pairs, dimension_names)):
    plt.figure(figsize=(8, 6))
    plt.scatter(max_radius_samples[:, dim_pair[0]], max_radius_samples[:, dim_pair[1]], 
                color='blue', label='Samples around Center R')
    plt.scatter(max_radius_R[dim_pair[0]], max_radius_R[dim_pair[1]], 
                color='red', marker='x', s=100, label='Center R')
    plt.title(f'2D Projection in {dim_names[0]} vs {dim_names[1]}')
    plt.xlabel(dim_names[0])
    plt.ylabel(dim_names[1])
    plt.legend()
    plt.grid(True)
    
    # Save the image to the specified directory
    save_path = f'/scratch/htc/btian/NNs_test/R_largest_radius_samples_{dim_names[0]}_{dim_names[1]}.png'
    plt.savefig(save_path)
    plt.show()

    print(f"Plot saved to: {save_path}")
