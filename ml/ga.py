import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost
from sklearn.svm import SVR
import argparse
import joblib
from ga_utils import generations
from tqdm import tqdm
import random
import torch
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


columns = ['Mg_meq_L', 'TOTP_mg_L', 'pH', 'TEMP_°C', 'TSS_mg_L', 'DO_mg_L', 'NH4N_mg_L', 'SO4_meq_L', 'Ca_meq_L', 'Cl_meq_L', 'NO32_mg_L', 'ALK_meq_L', label]
columns = list(set(columns))

# train_path = '../data/impute10/train/'
# test_path = '../data/impute10/test/'
# train_files = os.listdir(train_path)
# test_files = os.listdir(test_path)
# train_df = pd.concat(
#     [pd.read_csv(os.path.join(train_path, f), usecols=columns) for f in train_files])
# test_df = pd.concat(
#     [pd.read_csv(os.path.join(test_path, f), usecols=columns) for f in test_files])

train_df = pd.read_csv('../data/combined_0_to_nan_impute/train.csv', usecols=columns)
test_df = pd.read_csv('../data/combined_0_to_nan_impute/test.csv', usecols=columns)

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
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=1),
    xgboost.XGBRegressor(),
    SVR()
]


def acc_score():
    Score = pd.DataFrame({"Classifier": regression_models})
    j = 0
    mae = []
    mse = []
    rmse = []
    r2 = []
    for i in tqdm(models, desc='Training models without GA...'):
        model = i
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # metrics for regression
        mae.append(mean_absolute_error(y_test, predictions))
        mse.append(mean_squared_error(y_test, predictions))
        rmse.append(np.sqrt(mean_squared_error(y_test, predictions)))
        r2.append(r2_score(y_test, predictions))
        j = j+1
    Score["MAE"] = mae
    Score["MSE"] = mse
    Score["RMSE"] = rmse
    Score["r2-score"] = r2
    Score.sort_values(by="r2-score", ascending=False, inplace=True)
    Score.reset_index(drop=True, inplace=True)
    best_model_non_ga = Score.loc[0, 'Classifier']
    return Score, best_model_non_ga


# scores, best_model_non_ga = acc_score()
# print('Scores of all model without GA:')
# print(scores)
# print('Best model without GA:', best_model_non_ga)

# print('=============================================')



# Use the best non-GA model for GA feature selection
# logmodel = models[regression_models.index(best_model_non_ga)]

# model_name = 'RandomForestRegressor'

logmodel = models[regression_models.index(model_name)]

best_score = {
    "mae": 0,
    "mse": 0,
    "rmse": 0,
    "r2": 0,
}
best_chromo_all = []

# best_score_all = 0
# best_mse_all = 0

# best_model, best_mse_all, best_score_all, best_chromo_all = generations(best_mse_all, best_score_all, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, size=80, n_feat=X_train.shape[1], n_parents=64, mutation_rate=0.20, n_gen=5)

best_model, best_score, best_chromo_all = generations(best_score, best_chromo_all, logmodel, X_train, y_train, X_test, y_test, size=80, n_feat=X_train.shape[1], n_parents=64, mutation_rate=0.20, n_gen=5)



selected_columns = [col for col, flag in zip(usecols, best_chromo_all) if flag]

print(f'Original columns ({len(usecols)}): ', usecols.to_list())
print(f'Selected columns ({len(selected_columns)}): ', selected_columns)
print('Best MAE:', best_score['mae'])
print('Best MSE:', best_score['mse'])
print('Best RMSE:', best_score['rmse'])
print('Best R2 score:', best_score['r2'])
print('=============================================')

# Save the best model
joblib.dump(best_model, f'./results/{model_name}.pkl')


# save selected columns to txt file
with open(f'./results/{model_name}_selected_columns.txt', 'w') as f:
    for item in selected_columns:
        f.write("%s\n" % item)