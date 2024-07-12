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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
args = parser.parse_args()
optim = args.optim

label = 'COND_mS_m'

columns = ['Mg_meq_L', 'TOTP_mg_L', 'pH', 'TEMP_°C', 'TSS_mg_L', 'DO_mg_L',  'NH4N_mg_L', 'SO4_meq_L', 'Ca_meq_L', 'Cl_meq_L', 'NO32_mg_L', 'ALK_meq_L', label]

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


train_df = pd.read_csv('../../data/combined_0_to_nan_impute/train.csv', usecols=columns)
test_df = pd.read_csv('../../data/combined_0_to_nan_impute/test.csv', usecols=columns)

def get_data_loaders(features, labels, batch_size, shuffle=True):
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1).to(device)

    dataset = torch.utils.data.TensorDataset(features, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def plot_learning_curve(train_losses, train_r2_scores, val_losses, val_r2_scores):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax2.plot(train_r2_scores, label='Training $R^2$ Score', color='red')
    ax2.plot(val_r2_scores, label='Validation $R^2$ Score', color='green')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color='blue', fontsize=14)
    ax2.set_ylabel('$R^2$ Score', color='red', fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax1.legend(loc='lower left', fontsize=12)
    ax2.legend(loc='upper left', fontsize=12)
    plt.grid()
    # plt.title('Learning Curve')
    plt.savefig(f'./results/learning_curve_{optim}.png')
    plt.savefig(f'./results/learning_curve_{optim}.pdf')


# Training the model and recording the loss
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
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
        all_train_predictions = []

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
            all_train_predictions.extend(outputs.cpu().detach().numpy())

        avg_epoch_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        train_r2_scores.append(r2_score(all_train_labels, all_train_predictions))
        # if (epoch+1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, R^2: {train_r2_scores[-1]:.4f}')

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            all_val_labels = []
            all_val_predictions = []
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(outputs.cpu().detach().numpy())

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_r2_scores.append(r2_score(all_val_labels, all_val_predictions))

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Train score: {train_r2_scores[-1]:.4f}, Train loss: {avg_epoch_loss:.4f}, Val score: {val_r2_scores[-1]:.4f}, Val loss: {avg_val_loss:.4f}')

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                # torch.save(model.state_dict(), f'./models/best_model_{optim}.pth')

            if epoch - best_epoch >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    return train_losses, train_r2_scores, val_losses, val_r2_scores


def initilization_of_population(size, n_feat):
    population = []
    for _ in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool_)
        chromosome[:int(0.3*n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(best_model, best_score, best_chromo, population, selected_test_loader):
    scores = []

    train_labels = train_df[label].values
    test_labels = test_df[label].values

    for chromosome in tqdm(population, desc='Training models with GA...'):
        selected_columns = [columns[i] for i in range(len(columns)-1) if chromosome[i]]
        print('Selected columns:', selected_columns)

        train_features_selected = train_df[selected_columns].values
        test_features_selected = test_df[selected_columns].values

        train_labels = train_df[label].values
        test_labels = test_df[label].values


        train_features_selected, val_features_selected, train_labels, val_labels = train_test_split(
            train_features_selected, train_labels, test_size=0.2, random_state=40)

        scaler = StandardScaler()
        train_features_selected = scaler.fit_transform(train_features_selected)
        test_features_selected = scaler.transform(test_features_selected)
        val_features_selected = scaler.transform(val_features_selected)

        train_loader = get_data_loaders(train_features_selected, train_labels, batch_size=batch_size)
        val_loader = get_data_loaders(val_features_selected, val_labels, batch_size=batch_size, shuffle=False)
        test_loader = get_data_loaders(test_features_selected, test_labels, batch_size=batch_size, shuffle=False)


        model = ANN(input_dim=len(selected_columns), hidden_dim=hidden_dim, output_dim=output_dim).to(device)

        if optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        train_losses, train_r2_scores, val_losses, val_r2_scores = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

        predictions = []
        model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions).flatten()
        y_test = test_labels.flatten()

        scores.append(r2_score(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        print('R^2 score:', scores[-1], 'MSE:', mse, 'Chromosome:', chromosome)

        if scores[-1] > best_score['r2']:
            best_score['mae'] = mae
            best_score['mse'] = mse
            best_score['rmse'] = rmse
            best_score['r2'] = scores[-1]
            best_chromo = chromosome
            best_model = model
            plot_learning_curve(train_losses, train_r2_scores, val_losses, val_r2_scores)
            selected_test_loader = test_loader

        print('Best R^2 score:', best_score['r2'])


    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)

    return list(scores[inds][::-1]), list(population[inds][::-1]), best_score, best_chromo, best_model, selected_test_loader


def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0, len(pop_after_sel), 2):
        new_par = []
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i+1]
        new_par = np.concatenate(
            (child_1[:len(child_1)//2], child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate, n_feat):
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range):
            pos = randint(0, n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(size, n_feat, n_parents, mutation_rate, n_gen):
    selected_test_loader = None
    population_nextgen = initilization_of_population(size, n_feat)
    best_model = None
    best_score = {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}
    best_chromo = None

    for i in range(n_gen):
        print('Generation:', i+1)
        scores, pop_after_fit, best_score, best_chromo, best_model, selected_test_loader = fitness_score(best_model, best_score, best_chromo, population_nextgen, selected_test_loader)
        print('Best score in generation', i+1, ':', scores[0])

        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)

    return best_model, best_score, best_chromo, selected_test_loader

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out


def plot_results(y_true, y_pred):
    max_val = max(y_true.max(), y_pred.max())
    plt.figure(figsize=(5, 5))
    plt.plot([0, max_val], [0, max_val], color='black', linewidth=2, linestyle='-')
    sns.scatterplot(x=y_true, y=y_pred, color='red', alpha=0.5)
    plt.xlabel('Grounth truth', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.savefig(f'./results/actual_predictions_{optim}.png')
    plt.savefig(f'./results/actual_predictions_{optim}.pdf')


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode`

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
        print(f'MAE: {mae:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'R^2 score: {r2:.4f}')

        use_columns = columns[:-1]
        print('Selected features:', [use_columns[i] for i in range(len(best_chromo)) if best_chromo[i]])

        plot_results(y_true.flatten(), y_pred.flatten())

        # save predictions and true values to a csv file
        results = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
        results.to_csv(f'./results/results_{optim}.csv', index=False)

batch_size = 128
hidden_dim = 50
output_dim = 1
criterion = nn.MSELoss()
learning_rate = 0.001
num_epochs = 2000
patience = 50


# Genetic Algorithm
best_model, best_score, best_chromo, selected_test_loader = generations(size=80, n_feat=len(columns)-1, n_parents=64, mutation_rate=0.2, n_gen=5)

evaluate_model(best_model, selected_test_loader)
torch.save(best_model.state_dict(), f'./models/model_state_dict_{optim}.pth')
torch.save(best_model, f'./models/model_{optim}.pth')
torch.save(selected_test_loader, f'./test_loader/test_loader_{optim}.pth')
