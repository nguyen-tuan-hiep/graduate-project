import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import seaborn as sns


df = pd.read_csv('./preprocessing/combined_0_to_nan_impute.csv')

df = df.drop(columns=['MC', 'STATID', 'Y', 'M', 'D'], axis=1)


df['SDATE'] = pd.to_datetime(df['SDATE'])
df = df.set_index('SDATE')

# df = df.groupby('SDATE').mean()

df['Year'] = df.index.year
df['Month'] = df.index.month
df_monthly = df.groupby(['Year', 'Month']).mean()

# Reset index to get 'Year' and 'Month' back as columns
df_monthly.reset_index(inplace=True)

# Convert 'Year' and 'Month' back to a datetime index
df_monthly['SDATE'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(DAY=1))
df_monthly.set_index('SDATE', inplace=True)

# Drop 'Year' and 'Month' columns as they are no longer needed
df_monthly.drop(columns=['Year', 'Month'], inplace=True)

def plot_smoothed_relationship(df, feature, window_size=10, threshold=0.4):
    # high_corr_columns = corr_filtered[(corr_filtered[feature].abs() > threshold)].index.drop([feature])

    start_date = pd.to_datetime('1985-01-01')
    for col in df.columns:
        df_copy = df.copy()

        df_copy.index = pd.to_datetime(df_copy.index)
        corr_filtered = df_copy.corr()

        corr_value = corr_filtered.loc[feature, col]

        df_copy[f'{col}_MA'] = df_copy[col].rolling(window=window_size).mean()
        df_copy[f'{feature}_MA'] = df_copy[feature].rolling(window=window_size).mean()

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        # breakpoint()
        sns.lineplot(data=df_copy, x=df_copy.index, y=f'{feature}_MA', ax=ax1, label=f'{feature}', marker='o', color='r')
        sns.lineplot(data=df_copy, x=df_copy.index, y=f'{col}_MA', ax=ax2, label=f'{col}', marker='o', color='b')


        ax1.set_ylabel(f'{feature} Values', fontsize=15)
        ax2.set_ylabel(f'{col} Values', fontsize=15)

        # plt.title(f'Relationship of {feature} with {col}. Corr: {corr_value:.2f}')
        ax1.set_xlabel('Date', fontsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        locator = AutoDateLocator(minticks=5, maxticks=10)
        formatter = DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        plt.grid(True)
        ax1.set_xlim(left=pd.to_datetime(start_date), right=pd.to_datetime('2022-01-01'))
        ax1.legend(loc='upper left', fontsize=15)
        ax2.legend(loc='upper right', fontsize=15)
        plt.savefig(f'./results/{feature}_{col}.png')
        plt.savefig(f'./results/{feature}_{col}.pdf')


plot_smoothed_relationship(df_monthly, 'COND_mS_m')
