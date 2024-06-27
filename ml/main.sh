python regression_model_ga.py --model_name="RandomForestRegressor"
python regression_model_ga.py --model_name="GradientBoostingRegressor"
python regression_model_ga.py --model_name="SVR"
python regression_model_ga.py --model_name="XGBRegressor"

python test.py --model_name="RandomForestRegressor"
python test.py --model_name="GradientBoostingRegressor"
python test.py --model_name="SVR"
python test.py --model_name="XGBRegressor"

python best_model_detailed.py --model_name="RandomForestRegressor"
python best_model_detailed.py --model_name="GradientBoostingRegressor"
python best_model_detailed.py --model_name="SVR"
python best_model_detailed.py --model_name="XGBRegressor"