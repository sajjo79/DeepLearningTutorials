from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
# !pip install -q git+https://github.com/tensorflow/docs  install git hub. set its path. download docs-master and then use command
# pip install docs-master
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from  IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile
import os
#===========================================================
FEATURES = 28
N_VALIDATION = int(1000)
N_TRAIN = int(10000)
BUFFER_SIZE = int(10000)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
MAXEPOCHS=1000

def doConfig():
    logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)
    print(logdir)
    return logdir

def getdata():
    gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')
    ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")
    print(ds)
    return ds

def pack_row(*row):
  row = list(row)
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

def view_histogram(ds):
    packed_ds = ds.batch(10000).map(pack_row).unbatch()
    print("printing features")
    for features,label in packed_ds.batch(1000).take(1):
      print(features[0])
      #plt.hist(features.numpy().flatten(), bins = 101)
      #plt.show()
    print("----------------------")
    return packed_ds

def fetch_data(packed_ds):
    validate_ds = packed_ds.take(N_VALIDATION).cache()
    train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
    print(train_ds)
    validate_ds = validate_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
    return validate_ds,train_ds

def lr_scheduling():
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH * 1000,
        decay_rate=1,
        staircase=False)
    return lr_schedule

def get_optimizer(lr_schedule):
  return tf.keras.optimizers.Adam(lr_schedule)

def plot_lr(lr_schedule):
    step = np.linspace(0, 100000)
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step / STEPS_PER_EPOCH, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    _ = plt.ylabel('Learning Rate')
    #plt.show()

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def define_tiny_model():
    tiny_model = tf.keras.Sequential([
        layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return tiny_model

def define_medium_model():
    medium_model = tf.keras.Sequential([
        layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(64, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return medium_model

def define_large_model():
    large_model = tf.keras.Sequential([
        layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(512, activation='elu'),
        layers.Dense(512, activation='elu'),
        layers.Dense(512, activation='elu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return large_model

def define_L2_Large_model():
    l2_model = tf.keras.Sequential([
        layers.Dense(512, activation='elu',
                     kernel_regularizer=regularizers.l2(0.001),
                     input_shape=(FEATURES,)),
        layers.Dense(512, activation='elu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(512, activation='elu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(512, activation='elu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1, activation='sigmoid')
    ])
    return l2_model

def define_dropout_model():
    dropout_model = tf.keras.Sequential([
        layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
        layers.Dropout(0.5),
        layers.Dense(512, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return dropout_model

def define_combined_model():
    combined_model = tf.keras.Sequential([
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                     activation='elu', input_shape=(FEATURES,)),
        layers.Dropout(0.5),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                     activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                     activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                     activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return combined_model

def compile_and_fit(model, name, optimizer, train_ds,max_epochs=10000):
  model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

def plot_history(size_histories):
    plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
    plotter.plot(size_histories)
    #epoch=size_histories.epoch
    #bin_cr=size_histories.history['binary_crossentropy']
    #val_bin_cr = size_histories.history['val_binary_crossentropy']
    #plt.plot(epoch,bin_cr)
    #plt.plot(epoch, val_bin_cr)
    #plt.xlabel('Epoch')
    #plt.ylabel('binary_crossentropy')
    plt.ylim([0.5, 0.7])
    plt.show()

def plot_all(size_histories):
    plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
    plotter.plot(size_histories)
    a = plt.xscale('log')
    plt.xlim([5, max(plt.xlim())])
    plt.ylim([0.5, 0.7])
    plt.xlabel("Epochs [Log Scale]")
    plt.show()

def plot_regularizer_histories(regularizer_histories):
    plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
    plotter.plot(regularizer_histories)
    plt.ylim([0.5, 0.7])
    plt.show()

def plot_dropout_histories(regularizer_histories):
    plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
    plotter.plot(regularizer_histories)
    plt.ylim([0.5, 0.7])

if __name__=="__main__":
    size_histories = {}

    logdir=doConfig()
    ds=getdata()
    packed_ds=view_histogram(ds)
    validate_ds,train_ds=fetch_data(packed_ds)
    lr_schedule=lr_scheduling()
    optimizer=get_optimizer(lr_schedule)
    plot_lr(lr_schedule)

    small_model=define_tiny_model()
    size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small',optimizer,train_ds,MAXEPOCHS)
    plot_history(size_histories)

    medium_model=define_medium_model()
    size_histories['Medium'] = compile_and_fit(medium_model, "sizes/Medium",optimizer,train_ds,MAXEPOCHS)
    plot_history(size_histories)

    large_model = define_large_model()
    size_histories['Large'] = compile_and_fit(large_model, "sizes/large", optimizer, train_ds, MAXEPOCHS)
    plot_history(size_histories)

    plot_all(size_histories)

    regularizer_histories = {}
    l2_model=define_L2_Large_model()
    regularizer_histories['Small'] = size_histories['Small']
    regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2",optimizer, train_ds, MAXEPOCHS)
    plot_regularizer_histories(regularizer_histories)

    dropout_model=define_dropout_model()
    regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout",optimizer, train_ds, MAXEPOCHS)
    plot_regularizer_histories(regularizer_histories)

    combined_model=define_combined_model()
    regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined",optimizer, train_ds, MAXEPOCHS)
    plot_regularizer_histories(regularizer_histories)


