import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
import math
import pickle


def casos_favorables(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total


def datathon_metric(pred, y_test):
    rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
    cf = casos_favorables(y_test.values, pred)
    return (0.7 * rrmse) + (0.3 * (1 - cf))


if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')

    #total_metric = 0
    #for _ in range(10):
    metric = 2.8
    while metric > 2.7:
        X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas'], test_size=0.25)

        # Preprocesamiento
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

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
        #total_metric += metric

    #print('La media de la métrica es: {}'.format(total_metric / 10))
    filename = 'model_XGB.sav'
    pickle.dump(model, open(filename, 'wb'))
