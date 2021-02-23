# pylint: disable=no-member

import sys
import math
import pandas as pd

# Diccionarios auxiliaries para la corrección de los precios
precio_actual = {}
primer_precio = {}

# Establece los precios no especificados como el último valor indicado para el mismo id ('-1' si nunca se ha especificado)


def completar_precios(row):
    precio = float(row['precio'].replace(',', '.')) if isinstance(
        row['precio'], str) else row['precio']
    identificador = str(row['id'])
    if math.isnan(precio):
        precio = precio_actual.get(identificador, -1.0)
    else:
        precio_actual[identificador] = precio
        if primer_precio.get(identificador) == None:
            primer_precio[identificador] = precio
    return precio

# Establece los precios que quedan sin especificar como el precio indicado más cercano


def completar_primeros_precios(row):
    precio = row['precio']
    if precio == -1.0:
        identificador = str(row['id'])
        precio = primer_precio.get(identificador)
    return precio


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == 'base':
        print("base")
        df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
        df.drop_duplicates(inplace=True)
        df.to_csv(index=False, path_or_buf=sys.argv[1].replace(
            '.txt', '') + "_base.csv", sep='|')
    else:
        df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
        df.drop_duplicates(inplace=True)
        # Quitar la hora de 'fecha'
        df['fecha'] = df['fecha'].apply(lambda x: x.replace(' 0:00:00', ''))
        # Completar parámetro precio
        df['precio'] = df.apply(completar_precios, axis=1)
        df['precio'] = df.apply(completar_primeros_precios, axis=1)
        # Separación de fecha en 3 columnas
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['dia'] = pd.DatetimeIndex(df['fecha']).day
        df['mes'] = pd.DatetimeIndex(df['fecha']).month
        df['anyo'] = pd.DatetimeIndex(df['fecha']).year
        df.drop('fecha', axis=1, inplace=True)

        # Corregir valores vacios de 'categoria_dos'
        df['categoria_dos'] = df['categoria_dos'].apply(
            lambda x: 0 if math.isnan(x) else x)

        # One-hot encoding de 'estado'
        df = pd.concat([df, pd.get_dummies(pd.get_dummies(
            df['estado'], prefix='estado'))], axis=1).drop(['estado'], axis=1)

        # One-hot encoding de 'categoria_uno'
        df = pd.concat([df, pd.get_dummies(pd.get_dummies(
            df['categoria_uno'], prefix='categoria_uno'))], axis=1).drop(['categoria_uno'], axis=1)

        # One-hot encoding de 'dia_atipico'
        df = pd.concat([df, pd.get_dummies(pd.get_dummies(
            df['dia_atipico'], prefix='dia_atipico'))], axis=1).drop(['dia_atipico'], axis=1)

        df['antiguedad'].fillna(0, inplace=True)

        df.to_csv(index=False, path_or_buf=sys.argv[1].replace(
            '.txt', '') + ".csv", sep='|')
