# Author: Sajid Iqbal, Ph.D
# This tutorial is taken from https://www.tensorflow.org/tutorials/keras/classification
# and modified for better understanding
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#=====================================================================================
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def getdata():
    # Fashion MNIST dataset which contains 70,000 grayscale images of size (28,28) in 10 categories
    # 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network
    # more details about Fashion MNIST at https://www.kaggle.com/zalando-research/fashionmnist
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return train_images,train_labels,test_images,test_labels

def checkdata(train_images,train_labels,test_images,test_labels):
    print(train_images.shape,train_labels.shape)
    print(test_images.shape,test_labels.shape)

def visualize_data(train_images,train_labels):
    fig=plt.figure()
    cname=class_names[train_labels[5]] # train_lables has numeric indexs. Using that index, find the name of class
    fig.suptitle(cname)
    plt.imshow(train_images[5])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def preprocessdata(train_images,test_images):
    # Scale these values to a range of 0 to 1 before feeding them to the neural network model.
    # To do so, divide the values by 255.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    print(np.max(train_images),np.min(train_images)) # checking the range of rescaled data
    print(np.max(test_images),np.min(test_images))
    return train_images,test_images

def visualize_again(train_images,train_labels):
    # recheck that if the data is correct even after rescaling
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def build_model():
    # a model is a specification just like math formula. it only works when data is provided to it
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',                         # the optimizer that we use
                  loss='sparse_categorical_crossentropy',   # the loss function
                  metrics=['accuracy'])                     # performance metric that we want to calculate
    print(model.summary())
    return model

def train_model(model,train_images,train_labels):
    epochs=10
    history=model.fit(train_images, train_labels, epochs=epochs)
    print(history.history['loss'])
    print(history.history['accuracy'])
    return model, history

def evaluate_model(test_images,test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_loss,test_acc

def predict_model(model,test_images):
    predictions = model.predict(test_images)
    print(predictions[0])               # it will print ten values. Probability of 0th image in each class
    pclass=np.argmax(predictions[0])    # find the index of maximum in list of ten numbers
    print(class_names[pclass])          # print the class name of predicted class
    return predictions

def find_prediction_accuracy(predictions,test_labels):
    print(predictions.shape)                                        # 10000 predictions, each has ten probabilities (10000,10)
    pred_indexs=np.argmax(predictions,axis=1)                       # (10000,10) --> (10000,1) 10000 indexes of max val in each record
    correct_preds=np.sum(pred_indexs==test_labels)
    accuracy=correct_preds/len(test_labels)
    print(accuracy)

if __name__=="__main__":
    train_images,train_labels,test_images,test_labels=getdata()     # 60000 images and their 60000 labels
    checkdata(train_images,train_labels,test_images,test_labels)    # 10000 images and their 10000 labels
    visualize_data(train_images, train_labels)
    train_images,test_images=preprocessdata(train_images, test_images)
    visualize_again(train_images,train_labels)
    model=build_model()
    model,history=train_model(model, train_images, train_labels)
    predictions=predict_model(model,test_images)
    find_prediction_accuracy(predictions, test_labels)


