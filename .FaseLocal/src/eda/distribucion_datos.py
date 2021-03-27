import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import sys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
    
    item_counts = df["id"].unique()
    print("items distinitos: " + str(len(item_counts)))

    campana = df['campaña'].value_counts()
    campana = campana[1]/(campana[0]+campana[1])
    print("dias de campaña: " + str(campana*100))

    campana = df['estado_No Rotura'].value_counts()
    campana = campana[1]/(campana[0]+campana[1])
    print("estado no Rotura: " + str(campana*100))

    campana = df['estado_Transito'].value_counts()
    campana = campana[1]/(campana[0]+campana[1])
    print("estado Transito: " + str(campana*100))
        
    campana = df['dia_atipico_-1'].value_counts()
    campana = campana[1]/(campana[0]+campana[1])
    print("dia atípico -1: " + str(campana*100))
    
    campana = df['dia_atipico_0'].value_counts()
    campana = campana[1]/(campana[0]+campana[1])
    print("dia atípico 0: " + str(campana*100))
    
    campana = df['dia_atipico_1'].value_counts()
    campana = campana[1]/(campana[0]+campana[1])
    print("dia atípico 1: " + str(campana*100))

    for letra in list('ABCDEFGHIKLNO'):
        clase = 'categoria_uno_' + letra
        campana = df[clase].value_counts()
        campana = campana[1]/(campana[0]+campana[1])
        print(clase + ": " + str(campana*100))