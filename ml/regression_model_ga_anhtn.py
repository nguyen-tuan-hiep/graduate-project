import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
import xgboost
from sklearn.svm import SVR
import argparse
import joblib
from ga_utils import generations
from tqdm import tqdm
import random
import os
import numpy as np
import torch


def set_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(1234)


parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str)
args = parser.parse_args()
label = args.label
# parser.add_argument('--model_name', type=str)
# model_name = args.model_name


train_path = '../data/train/'
train_files = os.listdir(train_path)


columns = ['Mg_meq_L', 'TOTP_mg_L', 'pH', 'TEMP_°C', 'TSS_mg_L', 'DO_mg_L',
           'NH4N_mg_L', 'SO4_meq_L', 'Ca_meq_L', 'Cl_meq_L', 'NO32_mg_L', 'ALK_meq_L', label]


columns = list(set(columns))

train_df = pd.concat(
    [pd.read_csv(os.path.join(train_path, f), usecols=columns) for f in train_files])

test_path = '../data/test/'
test_files = os.listdir(test_path)
test_df = pd.concat(
    [pd.read_csv(os.path.join(test_path, f), usecols=columns) for f in test_files])

y_train = train_df[label]
y_test = test_df[label]
X_train = train_df.drop(columns=['COND_mS_m'])
X_test = test_df.drop(columns=['COND_mS_m'])
usecols = X_train.columns

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=usecols)
X_test = pd.DataFrame(X_test, columns=usecols)


regression_models = [
    'MLPRegressor',
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'AdaBoostRegressor',
    'XGBRegressor',
    'SVR'
]

models = [
    MLPRegressor(activation='relu', alpha=0.001,
                 batch_size='auto', beta_1=0.9),
    RandomForestRegressor(n_estimators=100),
    GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=1),
    AdaBoostRegressor(),
    xgboost.XGBRegressor(),
    SVR()
]


def acc_score():
    Score = pd.DataFrame({"Classifier": regression_models})
    j = 0
    mse = []
    r2 = []
    for i in tqdm(models, desc='Training models without GA...'):
        model = i
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # metrics for regression
        mse.append(mean_squared_error(y_test, predictions))
        r2.append(r2_score(y_test, predictions))
        j = j+1
    Score["MSE"] = mse
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


# Use the best non-GA model for GA feature selection
logmodel = models[regression_models.index(best_model_non_ga)]


chromo_df_bc, score_bc, mse_bc, best_models, best_chromosome = generations(
    logmodel, X_train, y_train, X_test, y_test, size=20, n_feat=X_train.shape[1], n_parents=16, mutation_rate=0.20, n_gen=3)

r2_best_ga = max(score_bc)
mse_best_ga = min(mse_bc)
best_model_ga = best_models[score_bc.index(r2_best_ga)]
# chromo_of_best_ga = chromo_df_bc[score_bc.index(r2_best_ga)]
chromo_of_best_ga = best_chromosome

selected_columns = [col for col, flag in zip(
    X_train.columns, chromo_of_best_ga) if flag]

print('Selected columns:', selected_columns)
print('Number of original columns:', len(X_train.columns))
print('Number of columns selected by GA:', len(selected_columns))
print('Best R2 score:', r2_best_ga)
print('Best MSE:', mse_best_ga)
print('=============================================')

print('Best model with GA:', best_model_ga)
# joblib.dump(best_model_ga, 'best_ga_model.pkl')

# Use the selected columns for predictions
X_test_selected = X_test[selected_columns]

# for model in models:
#     model.fit(X_train[selected_columns], y_train)
#     y_pred = model.predict(X_test_selected)
#     print('Model:', model)
#     print('R2:', r2_score(y_test, y_pred))
#     print('MSE:', mean_squared_error(y_test, y_pred))
#     print('=============================================')
# model = RandomForestRegressor(n_estimators=100)
# model.fit(X_train[selected_columns], y_train)
# y_pred = model.predict(X_test_selected)

# # Create a DataFrame with actual and predicted values
# results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
