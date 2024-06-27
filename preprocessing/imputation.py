
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# df = pd.read_csv('./combined.csv', usecols=['MC', 'STATID', 'SDATE', 'Y', 'M', 'D', 'TEMP_°C', 'pH', 'TSS_mg/L', 'COND_mS/m', 'Ca_meq/L', 'Mg_meq/L', 'Na_meq/L', 'K_meq/L', 'ALK_meq/L', 'Cl_meq/L', 'SO4_meq/L', 'NO32_mg/L', 'NH4N_mg/L', 'TOTP_mg/L', 'DO_mg/L', 'CODMN_mg/L'])

df = pd.read_csv('./combined_0_to_nan.csv')


# only keep 48 stations in 48_station.txt
# according with column 'STATID' in df
# with open('./48_stations.txt', 'r') as f:
#     stations = f.readlines()
# stations = [x.strip() for x in stations]
# df = df[df['STATID'].isin(stations)]



# df = df.replace(' ', np.nan)
# df = df.replace('\n', np.nan)
# df = df.replace('-', np.nan)
# df = df.replace('<0.4', np.nan)
# df = df.replace('<0.40', np.nan)
# df = df.replace('<0.001', np.nan)
# df = df.replace('', np.nan)
# df = df.replace('No water', np.nan)

# df['DO_mg/L'] = pd.to_numeric(df['DO_mg/L'], errors='coerce')

# df['TEMP_°C'] = df['TEMP_°C'].astype(float)
# df['Na_meq/L'] = df['Na_meq/L'].astype(float)
# df['K_meq/L'] = df['K_meq/L'].astype(float)
# df['Cl_meq/L'] = df['Cl_meq/L'].astype(float)
# df['CODMN_mg/L'] = df['CODMN_mg/L'].astype(float)
# df['DO_mg/L'] = df['DO_mg/L'].astype(float)
# df['TSS_mg/L'] = df['TSS_mg/L'].astype(float)


# drop outliers
# df = df[df['TEMP_°C'] < 32]
# df = df[df['pH'] > 6]
# df = df[df['TSS_mg/L'] < 120]
# df = df[df['COND_mS/m'] < 30]
# df = df[df['Ca_meq/L'] < 1.6]
# df = df[df['Na_meq/L'] < 1]
# df = df[df['Mg_meq/L'] < 0.8]
# df = df[df['ALK_meq/L'] < 3]
# df = df[df['Cl_meq/L'] < 1]
# df = df[df['K_meq/L'] < 0.2]
# df = df[df['SO4_meq/L'] < 0.7]
# df = df[df['NO32_mg/L'] < 0.6]
# df = df[df['NH4N_mg/L'] < 0.15]
# df = df[df['TOTP_mg/L'] < 0.25]
# df = df[df['CODMN_mg/L'] < 10]


cols = df.columns[6:]

# change to float for all cols
for col in cols:
    # print(col)
    df[col] = df[col].astype(float)

# df = df.dropna()

# df.to_csv('./combined_cleaned_no_impute.csv', index=False)



imp = IterativeImputer(estimator=RandomForestRegressor(), max_iter=5, random_state=42, verbose=2)

df_imputed = imp.fit_transform(df.iloc[:, 6:])


df_imputed = pd.DataFrame(df_imputed, columns=df.iloc[:, 6:].columns)


df_first_6 = df.iloc[:, :6]
df_first_6 = df_first_6.reset_index(drop=True)

df_imputed = pd.concat([df_first_6, df_imputed], axis=1)


df_imputed = df_imputed.round(3)

df_imputed.to_csv('./combined_0_to_nan_impute.csv', index=False)
