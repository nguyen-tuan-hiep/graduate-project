import torch
import torch.nn as nn
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
parser.add_argument('--data', type=int)
args = parser.parse_args()
optim = args.optim
data = args.data

label = 'COND_mS_m'

columns = ['Mg_meq_L', 'TOTP_mg_L', 'pH', 'TEMP_°C', 'TSS_mg_L', 'DO_mg_L', 'NH4N_mg_L', 'SO4_meq_L', 'Ca_meq_L', 'Cl_meq_L', 'NO32_mg_L', 'ALK_meq_L', 'COND_mS_m']

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)
path = '../data/combined_0_to_nan_impute/' if data == 1 else '../data/combined_0_to_nan_drop/'

print(path)

train_df = pd.read_csv(os.path.join(path, 'train.csv'), usecols=columns)
test_df = pd.read_csv(os.path.join(path, 'test.csv'), usecols=columns)

# Separate features and labels for training and testing
train_features = train_df.drop(columns=[label]).values
train_labels = train_df[label].values

test_features = test_df.drop(columns=[label]).values
test_labels = test_df[label].values

train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=40)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
val_features = scaler.transform(val_features)

batch_size = 128

def get_data_loaders(features, labels, batch_size, shuffle=True):
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1).to(device)

    dataset = torch.utils.data.TensorDataset(features, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        # self.fc4 = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        # self.fc5 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc3(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc4(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc5(out)
        return out

train_loader = get_data_loaders(train_features, train_labels, batch_size)
test_loader = get_data_loaders(test_features, test_labels, batch_size, shuffle=False)
val_loader = get_data_loaders(val_features, val_labels, batch_size, shuffle=False)

input_dim = len(columns) - 1
hidden_dim = 20
output_dim = 1
criterion = nn.MSELoss()
learning_rate = 0.001
num_epochs = 2000
patience = 50

model = ANN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

if optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training the model and recording the loss
train_losses = []
train_r2_scores = []
val_losses = []
val_r2_scores = []

best_val_loss = float('inf')
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    all_train_labels = []
    all_train_preds = []
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_train_loss += loss.item()
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(outputs.cpu().detach().numpy())

    avg_epoch_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    train_r2_scores.append(r2_score(all_train_labels, all_train_preds))

    model.eval()
    epoch_val_loss = 0
    all_val_labels = []
    all_val_preds = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(outputs.cpu().numpy())

    avg_epoch_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_epoch_val_loss)
    val_r2_scores.append(r2_score(all_val_labels, all_val_preds))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Train R^2: {train_r2_scores[-1]:.4f}, Val Loss: {avg_epoch_val_loss:.4f}, Val R^2: {val_r2_scores[-1]:.4f}')

    # Early stopping
    if avg_epoch_val_loss < best_val_loss:
        best_val_loss = avg_epoch_val_loss
        best_epoch = epoch
        best_model = model.state_dict()
        # test_score = r2_score(y_true_val, y_pred_val)

        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(features)
            y_true = labels.cpu().detach().numpy()
            y_pred = outputs.cpu().detach().numpy()
            test_score = r2_score(y_true, y_pred)
        # print(f"Saving best model at epoch {epoch + 1} with validation loss: {best_val_loss:.4f} and test R^2 score: {test_score:.4f}, R^2 score Val: {val_r2_scores[-1]:.4f}")
        torch.save(best_model, f'./results/ann_model_{optim}.pth')
    elif epoch - best_epoch > patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break


print("Training complete")

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
ax1.plot(train_losses, label='Training Loss', color='blue')
ax2.plot(train_r2_scores, label='Training $R^2$ Score', color='red')
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', color='blue', fontsize=14)
ax2.set_ylabel('$R^2$ Score', color='red', fontsize=14)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax1.legend(loc='lower left', fontsize=12)
ax2.legend(loc='upper left', fontsize=12)
plt.grid()
plt.savefig(f'./learning_curve_{optim}.png')
# plt.savefig(f'./results/learning_curve_{optim}.pdf')

# torch.save(model.state_dict(), f'./results/ann_model_{optim}.pth')

def plot_results(y_true, y_pred):
    max_val = max(y_true.max(), y_pred.max())
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred)
    plt.plot([0, max_val], [0, max_val], color='black', linewidth=2, linestyle='-')
    sns.scatterplot(x=y_true, y=y_pred, color='red')
    plt.xlabel('Grounth truth', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    # plt.savefig(f'./results/actual_predictions_{optim}.png')
    # plt.savefig(f'./results/actual_predictions_{optim}.pdf')

model.load_state_dict(torch.load(f'./results/ann_model_{optim}.pth'))
model.eval()  # Set the model to evaluation mode

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
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    plot_results(y_true, y_pred)

    # save predictions and true values to a csv file
    # results = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
    # results.to_csv(f'./results/results_{optim}.csv', index=False)
