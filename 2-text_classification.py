from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
#================================================================================
def check_settings():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

def getdata():
    # Split the training set into 60% and 40%, so we'll end up with 15,000 examples
    # for training, 10,000 examples for validation and 25,000 examples for testing.
    # Each example is a sentence representing the movie review and a corresponding label
    train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
    (train_data, validation_data), test_data = tfds.load(
        name="imdb_reviews",
        split=(train_validation_split, tfds.Split.TEST),
        as_supervised=True)

    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch.shape, train_labels_batch.shape)
    print(train_examples_batch)
    print(train_labels_batch)
    test_examples_batch, test_labels_batch = next(iter(train_data.batch(10)))
    print(test_examples_batch.shape, train_labels_batch.shape)
    print(test_examples_batch)
    print(test_labels_batch)
    return train_data,validation_data,test_data, train_examples_batch,train_labels_batch

def getTrainedModel(train_examples_batch):
    """
    Text embedding based on Swivel co-occurrence matrix factorization[1] with pre-built OOV.
    Maps from text to 20-dimensional embedding vectors.
    The module takes a batch of sentences in a 1-D tensor of strings as input.
    The module preprocesses its input by splitting on spaces.
    Vocabulary contains 20,000 tokens and 1 out of vocabulary bucket for unknown tokens.
    """
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],   #Keras layer that uses a TensorFlow Hub model to embed the sentences
                               dtype=tf.string, trainable=True)
    print(train_examples_batch[:2])                         # print first three examples
    print(hub_layer(train_examples_batch[:2]))              # print the embedding of first three examples
    return hub_layer

def build_model(hub_layer):
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model # return compiled model

def train_model(model,train_data,validation_data):
    # Uncomment the followings to see the details of data. However following commands will read first batch and the
    # training will start from next batch
    #train_examples,train_labels=next(iter(train_data.shuffle(10000).batch(512)))
    #print(train_examples.shape,train_labels.shape)
    #valid_examples, valid_labels = next(iter(validation_data.shuffle(10000).batch(512)))
    #print(valid_examples.shape,valid_labels.shape)

    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=20,
                        validation_data=validation_data.batch(512),
                        verbose=1)
    return model, history

def test_model(model,test_data):
    results = model.evaluate(test_data.batch(512), verbose=2)
    for name, value in zip(model.metrics_names, results):
      print("%s: %.3f" % (name, value))

if __name__=="__main__":
    check_settings()
    print("-------------------------------------------------------------------------------")
    train_data,validation_data,test_data, train_examples_batch,train_labels_batch=getdata()
    print("-------------------------------------------------------------------------------")
    hub_layer=getTrainedModel(train_examples_batch)
    print("-------------------------------------------------------------------------------")
    model=build_model(hub_layer)
    print("-------------------------------------------------------------------------------")
    model, history=train_model(model, train_data, validation_data)
    print("-------------------------------------------------------------------------------")
    print(len(history.history['loss']),history.history['loss'])
    print("-------------------------------------------------------------------------------")
    test_model(model, test_data)
