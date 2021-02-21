import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import math
import sys

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
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas'], test_size=0.3)

    # parameters = {'n_estimators': (100, 200, 300), 'criterion': ('mae', 'mse'), 'bootstrap': (
    #     True, False), 'oob_score': (True, False), 'warm_start': (True, False), 'verbose':[1], 'n_jobs':[-1]}

    parameters = {'n_estimators': (100, 200, 300), 'verbose':[2], 'n_jobs':[-1]}

    reg = RandomForestRegressor()
    clf = GridSearchCV(reg, parameters, scoring=mean_absolute_error)
    clf.fit(X_train, y_train)

    print(clf.cv_results_.keys())

    pred = clf.predict(X_test)

    pred = list(map(lambda x: round(x), pred))
    print('{:<24}   {}'.format("Pred", "True"))
    for i in range(15):
        print('{:<24}   {}'.format(pred[i], y_test.values[i]))

    rrmse = math.sqrt(mean_squared_error(y_test.values, pred)) / y_train.mean()
    cf = casos_favorables(y_test.values, pred)
    metric = (0.7 * rrmse) + (0.3 * (1 - cf))

    print('El mse: {}'.format(mean_squared_error(y_test, pred)))
    print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
    # print('El cf es: {}'.format(cf))
    # print('La métrica propia es: {}'.format(metric))
