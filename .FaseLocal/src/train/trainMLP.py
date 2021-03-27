import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import sys
import numpy as np


def casos_favorables(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total


def mlp():
    df = pd.read_csv(r'data/Modelar_UH2021_drop.csv', sep='|')
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas'], test_size=0.3)

    reg = MLPRegressor(verbose=True, alpha=0.00005, learning_rate='invscaling', max_iter=20)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    pred = list(map(lambda x: round(x), pred))

    rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
    cf = casos_favorables(y_test.values, pred)
    metric = (0.7 * rrmse) + (0.3 * (1 - cf))

    print('{:<24}   {}'.format("Pred", "True"))
    for i in range(15):
        print('{:<24}   {}'.format(pred[i], y_test.values[i]))

    print('El mse: {}'.format(mean_squared_error(y_test, pred)))
    print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
    print('El cf es: {}'.format(cf))
    print('La métrica propia es: {}'.format(metric))
