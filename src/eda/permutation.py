from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import sys
from sklearn.model_selection import train_test_split


def permutation():
    df = pd.read_csv(filepath_or_buffer=r'data/Modelar_UH2021_drop.csv', sep='|')
    X, y = df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas']
    rfr = RandomForestRegressor(n_estimators=10)
    rfr.fit(X, y)
    result = permutation_importance(rfr, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    print(result)