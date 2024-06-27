import os
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import joblib
import pickle
from random import randint
import random
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(42)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


label = 'COND_mS_m'

columns = ['Mg_meq_L', 'TOTP_mg_L', 'pH', 'TEMP_°C', 'TSS_mg_L', 'DO_mg_L',
           'NH4N_mg_L', 'SO4_meq_L', 'Ca_meq_L', 'Cl_meq_L', 'NO32_mg_L', 'ALK_meq_L', label]
columns = list(set(columns))

train_path = '../data/impute10/train/'
test_path = '../data/impute10/test/'
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
train_df = pd.concat(
    [pd.read_csv(os.path.join(train_path, f), usecols=columns) for f in train_files])
test_df = pd.concat(
    [pd.read_csv(os.path.join(test_path, f), usecols=columns) for f in test_files])

y_train = train_df[label]
y_test = test_df[label]
X_train = train_df.drop(columns=label)
X_test = test_df.drop(columns=label)

usecols = X_train.columns


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN regression model


class ANNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ANNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        # self.fc3 = nn.Linear(4, 2)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 16)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Dataset class
class MekongDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            output = model(X).squeeze()
            loss = criterion(output, y)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Training loop
def fit(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=80):
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device) # loss: mean squared error
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(
            f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    return history


# plot learning curve
def plot_learning_curve(history):
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./results/learning_curve_ann.pdf')
    plt.show()

# prediction
def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X).squeeze().cpu().numpy()


# # Model training
input_dim = X_train.shape[1]
model = ANNRegressor(input_dim).to(device)
train_dataset = MekongDataset(X_train, y_train)
val_dataset = MekongDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
history = fit(model, train_loader, val_loader, criterion, optimizer, device)
plot_learning_curve(history)
# Save model
# model_path = '../models/'
# model_name = 'mekong_ann.pth'




preds = predict(model, X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
mse = np.mean((y_test - preds)**2)
print(f"R2 score: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")



best_score_all = 0
best_mse_all = 0
best_chromo_all = []
best_model = None









exit()










def initilization_of_population(size, n_feat):
    population = []
    for _ in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool_)
        chromosome[:int(0.3*n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(best_model, best_mse_all, best_score_all, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, population):
    scores = []

    for chromosome in tqdm(population, desc='Training models with GA...'):
        model = logmodel

        # ANN model fitting
        selected_columns = [col for col, flag in zip(usecols, chromosome) if flag]
        X_train_selected = X_train[:, chromosome]
        X_test_selected = X_test[:, chromosome]

        train_dataset = MekongDataset(X_train_selected, y_train)
        val_dataset = MekongDataset(X_test_selected, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        input_dim = X_train_selected.shape[1]
        model = ANNRegressor(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = fit(model, train_loader, val_loader, criterion, optimizer, device)
        plot_learning_curve(history)

        # Prediction
        predictions = predict(model, X_test_selected)

        scores.append(r2_score(y_test, predictions))
        mse = mean_squared_error(y_test, predictions)

        print('Score:', scores[-1], 'MSE', mse, 'Chromosome:', chromosome)


        if scores[-1] > best_score_all:
            best_score_all = scores[-1]
            best_mse_all = mse
            best_chromo_all = chromosome
            best_model = model


    scores, population = np.array(scores), np.array(population)

    inds = np.argsort(scores)

    return list(scores[inds][::-1]), list(population[inds][::-1]), best_mse_all, best_score_all, best_chromo_all, best_model


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


def generations(best_mse_all, best_score_all, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, size, n_feat, n_parents, mutation_rate, n_gen):
    population_nextgen = initilization_of_population(size, n_feat)
    best_model = None
    for i in range(n_gen):
        scores, pop_after_fit, best_mse_all, best_score_all, best_chromo_all, best_model = fitness_score(best_model, best_mse_all, best_score_all, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, population_nextgen)
        print('Best score in generation', i+1, ':', scores[0])

        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)

    return best_model, best_mse_all, best_score_all, best_chromo_all



best_model, best_mse_all, best_score_all, best_chromo_all = generations(best_mse_all, best_score_all, best_chromo_all, model, X_train, y_train, X_test, y_test, size=20, n_feat=X_train.shape[1], n_parents=16, mutation_rate=0.20, n_gen=1)


selected_columns = [col for col, flag in zip(usecols, best_chromo_all) if flag]

print(f'Original columns ({len(usecols)}): ', usecols.to_list())
print(f'Selected columns ({len(selected_columns)}): ', selected_columns)
print('Best R2 score:', best_score_all)
print('Best MSE:', best_mse_all)
print('=============================================')

# Save best model
pickle.dump(best_model, open('./results/best_ann_ga_model.pkl', 'wb'))



# Save the selected columns
with open('./results/ann_selected_columns.txt', 'w') as f:
    f.write('\n'.join(selected_columns))
