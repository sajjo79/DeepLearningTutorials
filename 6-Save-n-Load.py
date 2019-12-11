from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
#===============================================================================================
def getdata():
    #60000,28,28         10000,28,28    10 categories
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    print(train_images.shape,train_labels.shape)
    print(test_images.shape,test_labels.shape)
    return train_images,train_labels,test_images,test_labels

def preprocess(train_images,train_labels,test_images,test_labels):
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    print(train_images.shape)
    print(test_images.shape)
    print(np.max(train_images),np.min(train_images))
    print(np.max(test_images), np.min(test_images))
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0 #(10000,784) (0-1)
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    print(np.max(train_images), np.min(train_images))
    print(np.max(test_images), np.min(test_images))
    print(train_images.shape)
    print(test_images.shape)
    return train_images,train_labels,test_images,test_labels

def create_n_compile_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  print(model.summary())
  return model

def train_model(model,train_images,train_labels):
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("save path of checkpoints ==> ",checkpoint_dir)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(x=train_images,             # Train the model with the new callback
              y=train_labels,
              epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])  # Pass callback to training
    return model,checkpoint_path

def evaluate_model(name,model,test_images,test_labels):
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(name,", accuracy: {:5.2f}%".format(100 * acc))

def load_weights(model,checkpoint_path):
    model.load_weights(checkpoint_path)
    return model

def train_model_2(model,train_images,train_labels):
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)
    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit(train_images,
              train_labels,
              epochs=50,
              callbacks=[cp_callback],
              validation_data=(test_images, test_labels),
              verbose=0)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    return model,latest


if __name__=="__main__":
    train_images,train_labels,test_images,test_labels=getdata()
    train_images,train_labels,test_images,test_labels=preprocess(train_images,train_labels,test_images,test_labels)
    model_1=create_n_compile_model()
    trained_model,checkpoint_path=train_model(model_1,train_images,train_labels)
    #
    model_2 = create_n_compile_model()
    evaluate_model("Untrained Model ",model_2, test_images, test_labels)
    model_2=load_weights(model_2, checkpoint_path)
    evaluate_model("Restored Model ", model_2, test_images, test_labels)
    #
    model_3=create_n_compile_model()
    trained_model_3,latest=train_model_2(model_3, train_images, train_labels)
    #
    model_4 = create_n_compile_model()
    model_4.load_weights(latest)
    evaluate_model("Restored Model ",model_4,test_images,test_labels)
    #
    model_5 = create_n_compile_model()
    model_5,checkpoint_path=train_model(model_5,train_images,train_labels)
    model_5.save_weights('./checkpoints/my_checkpoint')
    model_6 = create_n_compile_model()
    model_6.load_weights('./checkpoints/my_checkpoint')
    evaluate_model("Restored Model ",model_6,test_images,test_labels)
    #
    model_7 = create_n_compile_model()
    model_7.fit(train_images, train_labels, epochs=5)
    model_7.save('my_model.h5')
    new_model = tf.keras.models.load_model('my_model.h5')
    print(new_model.summary())
    evaluate_model("Restored Model",new_model,test_images,test_labels)

    # Assignment 2: display the training histories as we did in lecture 5.
    # improve the model by incorporating more layers, regularization and display its histories
    # save your trained model and weights, restore them also in your improved version of this program