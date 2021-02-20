import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
import sys


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


if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=sys.argv[1], sep='|')
    X, Y = df.loc[:, df.columns !=
                  'unidades_vendidas'], df['unidades_vendidas']

    model = keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    history = model.fit(
        X.values, Y.values,
        epochs=3, validation_split=0.25, verbose=2)

    plot_history(history)

    # rrmse = math.sqrt(mean_squared_error(y_test, pred)) / y_train.mean()
    # cf = casos_favorables(y_test.values, pred)
    # metric = (0.7 * rrmse) + (0.3 * (1 - cf))

    # print('El cf es: {}'.format(cf))
    # print('La métrica es: {}'.format(metric))
