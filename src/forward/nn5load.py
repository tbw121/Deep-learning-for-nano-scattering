import torch
import torch.nn as nn
import os

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = torch.sin(self.fc3(x))
        x = torch.sin(self.fc4(x))
        x = torch.sin(self.fc5(x))
        x = self.fc6(x)
        return x

# Load the test data
Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt').float().to('cuda')
Ytest = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt').float().to('cuda')

# Normalize Xtest using the same fixed min and max values
def normalize_tensor(tensor, min_val=140.0, max_val=280.0):
    return (tensor - min_val) / (max_val - min_val)

Xtest = normalize_tensor(Xtest)

# Define the save path
save_path = '/scratch/htc/btian/NNs_test'

# Load all models
num_models = 6
models = []
input_dim = Xtest.shape[1]
output_dim = Ytest.shape[1]

for rank in range(num_models):
    model = SimpleNN(input_dim, output_dim).to('cuda')
    model_path = os.path.join(save_path, f"model5p_{rank}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    models.append(model)

# Ensemble prediction function
def ensemble_predict(models, inputs):
    inputs = inputs.float().to('cuda')
    outputs = [model(inputs) for model in models]
    return torch.mean(torch.stack(outputs), dim=0)

# Evaluate ensemble model
ensemble_predictions = ensemble_predict(models, Xtest)
error = torch.sum(torch.abs(Ytest - ensemble_predictions)) / (Xtest.shape[0] * 5)
test_loss = error
print(f"Test Loss: {test_loss:.4f}")

# Test specific inputs
def test_specific_inputs(flux, radius):
    ensemble_predictionsr = ensemble_predict(models, radius)
    print("radius =", radius)
    print("flux =", flux)
    print("ensemble_predictionsr =", ensemble_predictionsr)
    errorr = torch.sum(torch.abs(flux - ensemble_predictionsr)) / (radius.shape[0] * 5)
    test_lossr = errorr
    print("test_lossr =", test_lossr)

# Example specific inputs
flux = torch.tensor([0.06175802, 0.11678328, 0.059668, 0.05023583, 0.10637244]).float().to('cuda')
radius = torch.tensor([193.28315992454932, 230.31789149244776, 280.243378716106, 171.4205764680762, 234.21772279028102, 116.6212940176413]).float().to('cuda')
test_specific_inputs(flux, radius)

flux = torch.tensor([5.23932570e-03, 7.67260261e-04, 7.49065999e-03, 2.79563742e-04, 8.98957656e-01]).float().to('cuda')
radius = torch.tensor([200.0, 200.0, 200.0, 220.0, 200.0, 200.0]).float().to('cuda')
test_specific_inputs(flux, radius)

flux = torch.tensor([0.02386115, 0.00408866, 0.00304581, 0.07206389, 0.6166373]).float().to('cuda')
radius = torch.tensor([200.0, 230.0, 200.0, 220.0, 180.0, 200.0]).float().to('cuda')
test_specific_inputs(flux, radius)

# Evaluate over a range of inputs
for i in range(-20, -1):
    print(i, "----iteration------------------------------------------")
    flux = Ytest[i]
    radius = Xtest[i]
    print("radius =", radius)
    print("flux =", flux)
    ensemble_predictionsr = ensemble_predict(models, radius)
    print("ensemble_predictionsr =", ensemble_predictionsr)
    errorr = torch.sum(torch.abs(flux - ensemble_predictionsr)) / (radius.shape[0] * 5)
    test_lossr = errorr
    print("test_lossr =", test_lossr)
