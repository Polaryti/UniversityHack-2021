import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state
import sys
import math
import csv


def __favorable_cases(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total


def __datetime_parser(datetime):
    datetime = datetime.split('-')
    return datetime[2] + '-' + datetime[1] + '-' + datetime[0]

def __datathon_metric(y_test, y_train, pred):
    rrmse = math.sqrt(mean_squared_error(
        y_test.values, pred)) / y_train.mean()
    cf = __favorable_cases(y_test.values, pred)
    return (0.7 * rrmse) + (0.3 * (1 - cf))


if __name__ == "__main__":
    modelar = pd.read_csv(
        r'data/Modelar_UH2021_drop.csv', sep='|')
    modelar = modelar.drop('estado_Rotura', axis=1)
    estimar = pd.read_csv(
        r'data/Estimar_UH2021_drop.csv', sep='|')
    estimar_data = pd.read_csv(
        r'data/Estimar_UH2021_base.csv', sep='|')

    score = math.inf
    rng = None
    while score > 2:
        rng = check_random_state(0)
        X_train, X_test, y_train, y_test = train_test_split(
            modelar.loc[:, modelar.columns != 'unidades_vendidas'], modelar['unidades_vendidas'], test_size=0.2, random_state=rng)

        model = RandomForestRegressor(
            n_estimators=150, min_samples_split=3, min_samples_leaf=2, n_jobs=-1, random_state=rng)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        pred = list(map(lambda x: round(x), pred))
        
        cf = __favorable_cases(y_test.values, pred)
        metric = __datathon_metric(y_test, y_train, pred)

        print('El mse: {}'.format(mean_squared_error(y_test, pred)))
        print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
        print('El cf es: {}'.format(cf))
        print('La métrica propia es: {}'.format(metric))
        print('Random state: {}'.format(rng))
        score = metric

    with open('res/Nevermore_debug_{}.txt'.format(rng), 'w') as debug:
        csvwriter = csv.writer(debug, delimiter='|')
        csvwriter.writerow(['predicted_label', 'true_label'])
        for i in range(len(pred)):
            csvwriter.writerow([pred[i], y_test.values[i]])

    with open('res/Nevermore_{}.txt'.format(rng), 'w') as csv_file:
        estimar_prediction = model.predict(estimar)
        csvwriter = csv.writer(csv_file, delimiter='|')
        csvwriter.writerow(['FECHA', 'ID', 'UNIDADES'])
        i = 0
        for est_pred in estimar_prediction:
            csvwriter.writerow(
                [__datetime_parser(estimar_data.iloc[i]['fecha']), int(estimar.iloc[i]['id']), round(est_pred)])
            i += 1
