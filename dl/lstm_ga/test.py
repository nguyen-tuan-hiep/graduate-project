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


test_loader = torch.load(f'./test_loader/test_loader_{optim}.pth', map_location=device)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = torch.load(f'./models/model_{optim}.pth', map_location=device)

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
