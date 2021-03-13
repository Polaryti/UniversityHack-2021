import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import sys
import numpy as np
import pickle


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


    reg = RandomForestRegressor(verbose=0, n_jobs=-1, n_estimators=100)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    '''
    importances = reg.feature_importances_
    std = np.std([tree.feature_importances_ for tree in reg.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    pred = list(map(lambda x : round(x), pred))
    print('{:<24}   {}'.format("Pred", "True"))
    for i in range(15):
        print('{:<24}   {}'.format(pred[i], y_test.values[i]))
    '''
    rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
    cf = casos_favorables(y_test.values, pred)
    metric = (0.7 * rrmse) + (0.3 * (1 - cf))
    '''
    print('El mse: {}'.format(mean_squared_error(y_test, pred)))
    print('El mae: {}'.format(mean_absolute_error(y_test, pred)))'''
    print('El cf es: {}'.format(cf))
    print('La métrica propia es: {}'.format(metric))

    
    filename = 'model_RF.sav'
    pickle.dump(reg, open(filename, 'wb'))
