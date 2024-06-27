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

# print the feature selection
with open(f'./results/{model_name}_selected_columns.txt', 'r') as f:
    columns = f.readlines()
    columns = [c.strip() for c in columns]
    print(columns)