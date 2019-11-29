from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
#===============================================================
def getdata():
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    print(dataset.columns)
    print(dataset.tail())
    print(dataset.isna().sum())
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    print(dataset['Origin'])
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='') # Convert categorical variable into dummy/indicator variables.
    print(dataset.tail())
    return dataset

def prepare_data(dataset):
    train_dataset = dataset.sample(frac=0.8,random_state=0) # Return a random sample of items from an axis of object.
    test_dataset = dataset.drop(train_dataset.index)        #Drop specified labels from rows or columns.
    print(train_dataset.shape)
    print(test_dataset.shape)
    #sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    #plt.show()

    train_stats = train_dataset.describe()
    test_stats=test_dataset.describe()
    print(train_stats)
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('MPG') # MPG is class
    test_labels = test_dataset.pop('MPG')
    return train_dataset,train_labels, test_dataset,test_labels,train_stats,test_stats

def norm(x,train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def preprocess_data(train_dataset,test_dataset,train_stats):
    normed_train_data = norm(train_dataset,train_stats)
    normed_test_data = norm(test_dataset,train_stats)
    return normed_train_data,normed_test_data

def build_model(train_dataset):
      print(len(train_dataset.keys()))
      model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
      ])

      optimizer = tf.keras.optimizers.RMSprop(0.001)

      model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
      return model

def predict_model(model,normed_train_data):
    example_batch = normed_train_data[:10]
    print(normed_train_data.shape,example_batch.shape)
    example_result = model.predict(example_batch)
    print(example_result)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def train_model(model,normed_train_data,train_labels):
    EPOCHS = 10#00
    history = model.fit(
      normed_train_data, train_labels,
      epochs=EPOCHS, validation_split = 0.2, verbose=0,
      callbacks=[PrintDot()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    print(hist.shape)
    return model,hist

def plot_history(hist):
  print(hist.shape)
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  epochs=hist['epoch']
  maes=hist['mae']
  val_mae=hist['val_mae']
  plt.plot(epochs,maes, label='Train Error')
  plt.plot(epochs,val_mae, label = 'Val Error')
  #plt.show()
  #plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  plt.show()
  print("done")

def train_model_early_stopping(model,normed_train_data,train_labels):
    EPOCHS = 1000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    return model,hist

def evaluate_model(normed_test_data, test_labels):
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

def predict_model2(model,normed_test_data,test_labels):
    test_predictions = model.predict(normed_test_data).flatten()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()

if __name__=="__main__":
    dataset=getdata()
    train_dataset, train_labels, test_dataset, test_labels, train_stats, test_stats=prepare_data(dataset)
    normed_train_data,normed_test_data=preprocess_data(train_dataset,test_dataset,train_stats)
    model=build_model(train_dataset)
    predict_model(model, normed_train_data) # prediction without training
    model,history=train_model(model, normed_train_data, train_labels)
    print(history.shape)
    plot_history(history)
    predict_model2(model, normed_test_data,test_labels)
    model,history=train_model_early_stopping(model, normed_train_data, train_labels)
    plot_history(history)
    evaluate_model(normed_test_data, test_labels)
    predict_model2(model, normed_test_data,test_labels)