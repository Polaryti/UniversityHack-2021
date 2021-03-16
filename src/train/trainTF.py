import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
import sys

from tensorflow.python.keras import activations


def casos_favorables(test, pred):
    rotura = 0
    total = len(test)
    for i in range(total):
        if pred[i] < test[i]:
            rotura += 1

    return (total - rotura) / total


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label='Val Error')
  plt.ylim([0, 5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label='Val Error')
  plt.ylim([0, 500])
  plt.legend()
  plt.show()


def tf():
    df = pd.read_csv(r'data/Modelar_UH2021_drop.csv', sep='|')
    X, Y = df.loc[:, df.columns !=
                  'unidades_vendidas'], df['unidades_vendidas']

    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

    history = model.fit(
        X.values, Y.values,
        epochs=2, validation_split=0.2, verbose=1)

    pred = model.predict(X.values)

    print(pred[:20])
    print(Y.values[:20])
    for i in range(20):
        print('{:<12}   {}'.format(pred[i], Y.values[i]))

    plot_history(history)

    # rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
    # cf = casos_favorables(y_test.values, pred)
    # metric = (0.7 * rrmse) + (0.3 * (1 - cf))

    # print('El cf es: {}'.format(cf))
    # print('La métrica es: {}'.format(metric))
