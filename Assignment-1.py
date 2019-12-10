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
import pandas as pd

# global variables
FEATURES = 28
N_VALIDATION = int(1000)
N_TRAIN = int(10000)
BUFFER_SIZE = int(10000)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE # 10000/500 --> 20
MAXEPOCHS=1000
logdir=""

def doConfig():
    logdir = pathlib.Path("E:\\PyCharmProjects\\TF_tutorials\\TF_Official_Tutorials\\tensorboard_logs")
    shutil.rmtree(logdir, ignore_errors=True)
    print(logdir)
    return logdir

def getdata():
    data = np.genfromtxt("E:\\HIGGS\\smallHIGGS.csv", delimiter=',')
    print("----------------------------------")
    labels=data[:,0]
    features=data[:,1:]
    print(data.shape)
    print(labels.shape)
    print(features.shape)
    train_labels=labels[0:10000]
    test_labels=labels[10000:11000]
    train_features=features[0:10000,:]
    test_features=features[10000:11000,:]
    print("train --> ",train_features.shape, train_labels.shape)
    print("test-->",test_features.shape,test_labels.shape)
    return train_features,train_labels,test_features,test_labels

def lr_scheduling():
    # When training a model, it is often recommended to lower the learning rate as the training progresses.
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH * 1000, #20,000
        decay_rate=1,
        staircase=False)

    step = np.linspace(0, 100000)   # [0.  2040.81632653   4081.63265306 ...]
    # initial_learning_rate / (1 + decay_rate * step / decay_step)
    # 0.001/(1+1*0/20000)
    lr = lr_schedule(step) # [0.001      0.00090741 0.00083051 0.00076563 0.00071014 0.00066216 ...
    return lr,step,lr_schedule

def plot_lr(lr,step):
    x_series=step / STEPS_PER_EPOCH
    y_series=lr
    plt.plot(x_series, y_series)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    _ = plt.ylabel('Learning Rate')
    plt.show()

def get_optimizer(lr_schedule):
  return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name,logdir):
  return [
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def define_tiny_model():
    tiny_model = tf.keras.Sequential([
        # input layer with 28 features
        layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return tiny_model

def compile_and_fit(model, name, optimizer, train_ds,validate_ds,max_epochs=10000):
  model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

  model.summary()

  history = model.fit(
    x=train_ds[0],
    y=train_ds[1],
    batch_size=500,
    epochs=max_epochs,
    steps_per_epoch = STEPS_PER_EPOCH,
    validation_data=validate_ds,
    callbacks=get_callbacks(name,logdir),
    verbose=1)
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

if __name__=="__main__":
    size_histories={}
    logdir=doConfig()
    train_x,train_y,test_x,test_y=getdata()
    lr,step,lr_schedule=lr_scheduling()
    plot_lr(lr,step)
    optimizer=get_optimizer(lr_schedule)
    small_model = define_tiny_model()
    train_ds=(train_x,train_y)
    validate_ds=(test_x,test_y)
    # model, name, optimizer, train_ds,validate_ds,max_epochs=10000
    size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small', optimizer, train_ds,validate_ds, MAXEPOCHS)
    plot_history(size_histories)
