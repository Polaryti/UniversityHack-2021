import csv
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


def date_parser(fecha):
    fecha = fecha.split('/')
    return f'{fecha[2]}-{fecha[0]}-{fecha[1]}'

modelar = pd.read_csv(r'data/Modelar_UH2021.txt', sep='|', low_memory=False)
# "Estimar" dataframe has not samples with "estado" = "Rotura" and that column is
# converted to one-hot vector so it must be dropped
modelar = modelar[['fecha', 'unidades_vendidas']]
modelar.set_index('fecha')
modelar['fecha'] = modelar['fecha'].apply(date_parser)

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=52,center=False).mean() 
    rolstd = timeseries.rolling(window=52,center=False).std()    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(modelar['unidades_vendidas'])