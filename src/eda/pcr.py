from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

def casos_favorables(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total

df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'unidades_vendidas'], df['unidades_vendidas'], test_size=0.25)

pcr = make_pipeline(StandardScaler(), PCA(
    n_components=1), RandomForestRegressor())
pcr.fit(X_train, y_train)
pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(pca.transform(X_test), y_test, alpha=.3, label='ground truth')
axes[0].scatter(pca.transform(X_test), pcr.predict(X_test), alpha=.3,
                label='predictions')
axes[0].set(xlabel='',
            ylabel='y', title='PCR / PCA')
axes[0].legend()
axes[1].scatter(pls.transform(X_test), y_test, alpha=.3, label='ground truth')
axes[1].scatter(pls.transform(X_test), pls.predict(X_test), alpha=.3,
                label='predictions')
axes[1].set(xlabel='',
            ylabel='y', title='PLS')
axes[1].legend()

pred = pcr.predict(X_test)
rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
cf = casos_favorables(y_test.values, pred)
metric = (0.7 * rrmse) + (0.3 * (1 - cf))
print('El mse: {}'.format(mean_squared_error(y_test, pred)))
print('El mae: {}'.format(mean_absolute_error(y_test, pred)))
print('El cf es: {}'.format(cf))
print('La métrica propia es: {}'.format(metric))

plt.tight_layout()
plt.show()
