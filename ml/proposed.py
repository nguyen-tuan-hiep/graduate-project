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

model = joblib.load(f'./results/{model_name}_gridsearch.pkl')

with sklearn.config_context(print_changed_only=False):
    print(model)

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


y_pred = model.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))


# print the feature selection
with open(f'./results/{model_name}_selected_columns.txt', 'r') as f:
    columns = f.readlines()
    columns = [c.strip() for c in columns]
    print(columns)