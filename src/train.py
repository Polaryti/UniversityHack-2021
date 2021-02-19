import pandas as pd
from sklearn import linear_model
import sys

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
    reg = linear_model.LinearRegression()
    reg.fit(df.loc[:, df.columns != 'precio'], df['precio'])
    reg.coef_