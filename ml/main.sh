# baseline
python regressor.py --data=drop

# missforest
python regressor.py --data=impute

# GA
python ga.py --model_name=RandomForestRegressor
python ga.py --model_name=GraidentBoostingRegressor
python ga.py --model_name=SVR
python ga.py --model_name=XGBRegressor

# Tuning
python tuning.py --model_name=RandomForestRegressor
python tuning.py --model_name=GraidentBoostingRegressor
python tuning.py --model_name=SVR
python tuning.py --model_name=XGBRegressor

python proposed.py --model_name=RandomForestRegressor
python proposed.py --model_name=GraidentBoostingRegressor
python proposed.py --model_name=SVR
python proposed.py --model_name=XGBRegressor
