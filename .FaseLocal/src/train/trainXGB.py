import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
import math

def casos_favorables(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total

def xgb_reg():
    df = pd.read_csv(r'data/Modelar_UH2021_drop.csv', sep='|')

    X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas'], test_size=0.25)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    param = {'objective': 'reg:tweedie'}
    model = xgb.train(param, dtrain)
    pred = model.predict(dtest)
    pred = list(map(lambda x: round(x), pred))

    print('{:<24}   {}'.format("Pred", "True"))
    for i in range(15):
        print('{:<24}   {}'.format(pred[i], y_test.values[i]))

    rrmse = math.sqrt(mean_squared_error(y_test.values, pred)) / y_train.mean()
    cf = casos_favorables(y_test.values, pred)
    metric = (0.7 * rrmse) + (0.3 * (1 - cf))

    print('El mse: {}'.format(mean_squared_error(y_test, pred)))
    print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
    print('El cf es: {}'.format(cf))
    print('La métrica propia es: {}'.format(metric))
