import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import math

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
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test) 

    for dep in [8,9,10,11,12,13,14,15,16,17,18]:
        cf_total = 0
        metrica_total = 0
        for _ in range(5):
            param = {'max_depth':dep, 'eta':1, 'objective':'reg:tweedie' }
            model = xgb.train(param, dtrain)
            pred = model.predict(dtest)
            '''
            print('{:<24}   {}'.format("Pred", "True"))
            for i in range(15):
                print('{:<24}   {}'.format(pred[i], y_test.values[i]))
            '''
            rrmse = math.sqrt(mean_squared_error(y_test.values, pred)) / y_train.mean()
            cf = casos_favorables(y_test.values, pred)
            metric = (0.7 * rrmse) + (0.3 * (1 - cf))
            cf_total += cf
            metrica_total += metric
            #print('El mse: {}'.format(mean_squared_error(y_test, pred)))
            #print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
        
        cf_total /= 5
        metrica_total /= 5
        print('depth: {}'.format(dep))
        print('El cf es: {}'.format(cf))
        print('La métrica propia es: {}'.format(metric))
        print('El cf es: {}'.format(cf_total))
        print('La métrica propia es: {}'.format(metrica_total))