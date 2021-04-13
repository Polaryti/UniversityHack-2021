import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import sys
import numpy as np
import pickle
import csv
from sklearn.preprocessing import StandardScaler


def casos_favorables(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total

def __datetime_parser(datetime):
    datetime = datetime.split('-')
    return datetime[2] + '-' + datetime[1] + '-' + datetime[0]


df_modelar = pd.read_csv('/home/mario/Documents/others/UniversityHack-2021/FaseNacional/data/Modelar_UH2021_drop.csv', sep='|')
df_estimar = pd.read_csv('/home/mario/Documents/others/UniversityHack-2021/FaseNacional/data/Estimar_UH2021_drop.csv', sep='|')
df_modelar.drop(columns=['estado_Rotura'], inplace=True)


model_dict = {}
total_cf = 0
total_metrica = 0
for index, row in df_modelar.iterrows():
    if row['id'] not in model_dict:
        df_aux = df_modelar[df_modelar['id'] == row['id']]
        X_train, X_test, y_train, y_test = train_test_split(
            df_aux.loc[:, df_aux.columns != 'unidades_vendidas'], df_aux['unidades_vendidas'], test_size=0.25)
        
        reg = RandomForestRegressor(verbose=0, n_jobs=-1, n_estimators=150, min_samples_split=3, min_samples_leaf=2)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)

        rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
        cf = casos_favorables(y_test.values, pred)
        metric = (0.7 * rrmse) + (0.3 * (1 - cf))

        # print(f"({row['id']}) CF: {cf}")
        total_cf += cf
        # print(f"({row['id']}) Métrica: {metric}")
        total_metrica += metric
        
        model_dict[row['id']] = reg

print(f"CF media: {total_cf / len(model_dict)}")
print(f"Metrica media: {total_metrica / len(model_dict)}")

with open('Nevermore_res.txt', 'w') as csv_file:
    csvwriter = csv.writer(csv_file, delimiter='|')
    csvwriter.writerow(['FECHA', 'ID', 'UNIDADES'])
    
    for index, row in df_estimar.iterrows():
        aux = pd.DataFrame(columns=df_estimar.columns)
        aux.loc[0] = row
        est_pred = model_dict[row['id']].predict(aux)
        csvwriter.writerow([__datetime_parser(df_estimar.iloc[index]['fecha']), int(df_estimar.iloc[index]['id']), round(est_pred)])

        