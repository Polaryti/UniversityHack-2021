import sys
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')
    df.drop_duplicates(inplace=True)
    # Quitar la hora de 'fecha'
    df['fecha'] = df['fecha'].apply(lambda x: x.replace(' 0:00:00', ''))
    # One-hot encoding de 'estado'
    df = pd.concat([df, pd.get_dummies(pd.get_dummies(df['estado'], prefix='estado'))], axis=1).drop(['estado'],axis=1)
    df.to_csv(index = False, path_or_buf= sys.argv[1].replace('.txt', '') + ".csv", sep='|')