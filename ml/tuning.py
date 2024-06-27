
from tqdm import tqdm
import random
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import sklearn
import argparse

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
model_name = args.model_name



label = 'COND_mS_m'


with open(f'./results/{model_name}_selected_columns.txt', 'r') as f:
    columns = f.readlines()
    columns = [c.strip() for c in columns]

columns.append(label)

columns = list(set(columns))

train_df = pd.read_csv('../data/combined_0_to_nan_impute/train.csv', usecols=columns)
test_df = pd.read_csv('../data/combined_0_to_nan_impute/test.csv', usecols=columns)


y_train = train_df[label]


y_test = test_df[label]
X_train = train_df.drop(columns=label)
X_test = test_df.drop(columns=label)

usecols = X_train.columns
print(f'Selected columns by GA: {usecols.to_list()}')


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=usecols)
X_test = pd.DataFrame(X_test, columns=usecols)


model = joblib.load(f'./results/{model_name}.pkl')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

# exit()


with sklearn.config_context(print_changed_only=False):
    print(model)

# exit()
if model_name == 'RandomForestRegressor' or model_name == 'GradientBoostingRegressor':
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 20, 30, 40],
        'min_samples_split': [1, 2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    }

elif model_name == 'SVR':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf'],
    }

elif model_name == 'XGBRegressor':
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'loss': ['ls', 'lad', 'huber'],
    }


model_gridsearch = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

model_gridsearch.fit(X_train, y_train)
print('Best params:', model_gridsearch.best_params_)

# y_pred_gridsearch = model_gridsearch.predict(X_test)
model_gridsearch = model_gridsearch.best_estimator_
y_pred_gridsearch = model_gridsearch.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred_gridsearch))
print('MSE:', mean_squared_error(y_test, y_pred_gridsearch))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_gridsearch)))
print('R2:', r2_score(y_test, y_pred_gridsearch))

# Use the best hyperparameters and apply the model

joblib.dump(model_gridsearch, f'./results/{model_name}_gridsearch.pkl')

# Save the predictions and the actual values to csv file
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gridsearch})
df.to_csv(f'./results/{model_name}_predictions.csv', index=False)