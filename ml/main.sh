# baseline
python regressor.py --data=drop

# misforest
python regressor.py --data=impute

# GA
python proposed.py --model_name=RandomForestRegressor
python proposed.py --model_name=GraidentBoostingRegressor
python proposed.py --model_name=SVR
python proposed.py --model_name=XGBRegressor
