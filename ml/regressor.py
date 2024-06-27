import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost
from sklearn.svm import SVR
import argparse
import joblib
from tqdm import tqdm
import random
import torch

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
parser.add_argument('--data', type=str)
args = parser.parse_args()
data = args.data

label = 'COND_mS_m'



columns = ['Mg_meq_L', 'TOTP_mg_L', 'pH', 'TEMP_°C', 'TSS_mg_L', 'DO_mg_L',
           'NH4N_mg_L', 'SO4_meq_L', 'Ca_meq_L', 'Cl_meq_L', 'NO32_mg_L', 'ALK_meq_L', label]
columns = list(set(columns))

if data == 'impute':
    path = '../data/combined_0_to_nan_impute'
elif data == 'drop':
    path = '../data/combined_0_to_nan_drop'

train_df = pd.read_csv(f'{path}/train.csv', usecols=columns)
test_df = pd.read_csv(f'{path}/test.csv', usecols=columns)


y_train = train_df[label]
y_test = test_df[label]
X_train = train_df.drop(columns=label)
X_test = test_df.drop(columns=label)

usecols = X_train.columns

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=usecols)
X_test = pd.DataFrame(X_test, columns=usecols)


regression_models = [
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'XGBRegressor',
    'SVR'
]

models = [
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    xgboost.XGBRegressor(),
    SVR()
]


def acc_score():
    Score = pd.DataFrame({"Classifier": regression_models})
    j = 0
    mse = []
    rmse = []
    mae = []
    r2 = []
    for i in tqdm(models, desc='Training models without GA...'):
        model = i
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # metrics for regression
        mse.append(mean_squared_error(y_test, predictions))
        r2.append(r2_score(y_test, predictions))
        mae.append(mean_absolute_error(y_test, predictions))
        rmse.append(np.sqrt(mean_squared_error(y_test, predictions)))
        j = j+1
    Score["MAE"] = mae
    Score["MSE"] = mse
    Score["RMSE"] = rmse
    Score["r2-score"] = r2
    Score.sort_values(by="r2-score", ascending=False, inplace=True)
    Score.reset_index(drop=True, inplace=True)
    best_model_non_ga = Score.loc[0, 'Classifier']
    return Score, best_model_non_ga


scores, best_model_non_ga = acc_score()
print('Scores of all model without GA:')
print(scores)
print('Best model without GA:', best_model_non_ga)

print('=============================================')
