import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'precio'], df['precio'], test_size=0.2)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    print(mean_squared_error(y_test, pred))