from __future__ import absolute_import, division, print_function, unicode_literals
#  __future__ module is used to make functionality available in the current version of
#  Python even though it will only be officially introduced in a future version.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# global variables
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

def tf_info():
    print(tf.__version__)

def getData():
    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
    with np.load(path) as data:
      train_X = data['x_train']
      train_Y = data['y_train']
      test_X = data['x_test']
      test_Y = data['y_test']
    return train_X,train_Y,test_X,test_Y

def getDatasets(train_X,train_Y,test_X,test_Y):
    train_DS = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    test_DS = tf.data.Dataset.from_tensor_slices((test_X, test_Y))
    train_DS = train_DS.shuffle(SHUFFLE_BUFFER_SIZE) #representing the number of elements from this dataset from which the new dataset will sample.
    train_DS=train_DS.batch(BATCH_SIZE) # sample data from shuffled data
    test_DS = test_DS.batch(BATCH_SIZE)
    return train_DS,test_DS

def visualize_data(train_DS):
    print(train_DS)
    ds=train_DS.take(1)
    imgs,labels=ds.element_spec
    print(imgs.shape,labels.shape)
    for img,lbl in ds:
        print(img.shape,lbl.shape)
        plt.imshow(img[0]),plt.show()
        print(lbl[0])


def build_n_compile():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    print(model.summary())
    return model

def train_model(model,train_DS):
    model.fit(train_DS, epochs=10)
    return model

def evaluate_model(model,test_X):
    preds=model.evaluate(test_X)
    return preds

def findAccuracy(preds,test_Y):
    # to do
    pass

def draw_loss_accuracy(preds,test_Y):
    # to do
    pass

def visualize_preds(test_DS,preds):
    #to do
    print(test_DS)
    print(preds)



if __name__=="__main__":
    train_X,train_Y,test_X,test_Y=getData()
    train_DS,test_DS=getDatasets(train_X, train_Y, test_X, test_Y)
    visualize_data(train_DS)
    # model=build_n_compile()
    # model=train_model(model,train_DS)
    # preds=evaluate_model(model,test_DS)
    # visualize_preds(test_DS,preds)
    #Assignments, introduce the following features in your network
    # learning schedule, regularization, drop out, larger models
    # use early stopping criteria to stop the training process when it becomes smooth