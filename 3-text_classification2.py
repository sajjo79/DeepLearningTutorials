from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
import numpy as np


# ====================================================
def get_and_test_data():
    # different programs may contain different ways of retrieving data
    # you need not worry about it as these are very specific commands for TensorFlow tutorials
    # for your own problem, you will write your own code to fetch data from hard disk or online
    # we encode the text strings (convert text string to numbers) so that neural network can process it
    (train_data, test_data), info = tfds.load(
        'imdb_reviews/subwords8k',                  # Use the version pre-encoded with an ~8k vocabulary.
        split=(tfds.Split.TRAIN, tfds.Split.TEST),  # Return the train/test datasets as a tuple.
        as_supervised=True,                         # Return (example, label) pairs from the dataset (instead of a dictionary).
        with_info=True)                             # Also return the 'info' structure.
    print(train_data)
    print(test_data)
    encoder = info.features['text'].encoder         # the object that encodes text string into numbers
    print('Vocabulary size: {}'.format(encoder.vocab_size)) # number of words in vocabulary collection
    print("done")

    sample_string = 'Hello TensorFlow.'
    encoded_string = encoder.encode(sample_string)
    print('Encoded string is {}'.format(encoded_string)) # example string that is encoded by encoder object

    original_string = encoder.decode(encoded_string)
    print('The original string: "{}"'.format(original_string)) # decoding the encoded string

    if(original_string == sample_string):             # checking whether both strings are equal
        print("both strings are equal")
    else:
        print("original and decoded strings are not equal")
    for ts in encoded_string:
        print('{} ----> {}'.format(ts, encoder.decode([ts]))) # prints each word and its corresponding encoding

    for train_example, train_label in train_data.take(1):   # take first record i.e. review and label
        print('Encoded text:', train_example[:10].numpy())  # print first 10 entries of encoded text (10 numbers)
        print('Label:', train_label.numpy())                # print label after converting it into numpy array
        original_string = encoder.decode(train_example[:10])# decode first 10 encoded entries
        print(original_string)                              # print decoded entries as text

    return train_data,test_data,info,encoder


def prepare_data(train_data, test_data):
    # train_batches and test_batches are specifications of data. These do not contain actual data
    BUFFER_SIZE = 1000
    print(train_data)                                       # Analyze the train data
    print(train_data.output_shapes)
    train_batches = (train_data
                    .shuffle(BUFFER_SIZE)                   # read 1000 records and shuffle them
                    .padded_batch(32, train_data.output_shapes)) # get batch of 32 reviews. Each review string is of different length
                                                                 # so padd string with zeros to make all strings of equal length
    test_batches = (
        test_data
            .padded_batch(32, train_data.output_shapes))
    print(train_batches)
    for example_batch, label_batch in train_batches.take(2):
        print("Batch shape:", example_batch.shape) # (batch_size, sequence_length)
        print("label shape:", label_batch.shape)
    return train_batches, test_batches


def build_model(encoder):
    model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation='sigmoid')])

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_batches, test_batches):
    history = model.fit(train_batches,
                        epochs=10,
                        validation_data=test_batches,
                        validation_steps=30)
    return model, history


def evaluate_model(model, test_batches):
    loss, accuracy = model.evaluate(test_batches)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


def plot_stats(history):
    history_dict = history.history
    history_dict.keys()

    import matplotlib.pyplot as plt

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


if __name__ == "__main__":
    train_data,test_data,info,encoder  = get_and_test_data()
    train_batches, test_batches=prepare_data(train_data, test_data)
    model=build_model(encoder)
    model, history=train_model(model, train_batches, test_batches)
    evaluate_model(model, test_batches)
    plot_stats(history)