import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import argparse
from sklearn.metrics import r2_score

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(40)

parser = argparse.ArgumentParser()
parser.add_argument('--optim', type=str)
parser.add_argument('--data', type=str)
args = parser.parse_args()
optim = args.optim
data = args.data

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)
if data == 'impute':
    path = '../../data/combined_0_to_nan_impute/'
elif data == 'drop':
    path = '../../data/combined_0_to_nan_drop/'

print(path)

test_loader = torch.load(f'{data}/test_loader/test_loader_{optim}.pth')

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = torch.load(f'{data}/models/model_{optim}.pth')

# input_dim = 12
# hidden_dim = 20
# output_dim = 1

# model = ANN(input_dim, hidden_dim, output_dim).to(device)

# # Load model state_dict
# model.load_state_dict(torch.load(f'{data}/models/model_state_dict_{optim}.pth'))

# evaluate the model on test_loader
model.eval()
with torch.no_grad():  # No need to compute gradients for validation
    y_true = []
    y_pred = []

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate R^2 score
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R^2 score: {r2:.4f}')
