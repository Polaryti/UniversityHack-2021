import pandas as pd
from sklearn.kernel_ridge import KernelRidge
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


if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas'], test_size=0.3)

    reg = KernelRidge()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
    cf = casos_favorables(y_test.values, pred)
    metric = (0.7 * rrmse) + (0.3 * (1 - cf))

    print('El mse: {}'.format(mean_squared_error(y_test, pred)))
    print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
    print('El cf es: {}'.format(cf))
    print('La métrica propia es: {}'.format(metric))
