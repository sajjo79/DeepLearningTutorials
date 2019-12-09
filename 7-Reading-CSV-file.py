# importing libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import csv
import pandas as pd

# global variables
LABEL_COLUMN = 'survived'
LABELS = [0, 1]
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']


def doConfig():
    np.set_printoptions(precision=3, suppress=True)

def getdatafiles():
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    print(train_file_path)
    return train_file_path,test_file_path

def inspect_csv(train_file_path,test_file_path):
    with open(train_file_path, 'rt') as f:
        data = csv.reader(f)
        for row in data:
            print(row)

def inspect_pandas(train_file_path,test_file_path):
    result = pd.read_csv(train_file_path)
    print(result)

def get_dataset(file_path, **kwargs):
    # API documentation path  https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset
    dataset = tf.data.experimental.make_csv_dataset(
          file_path,
          batch_size=5, # Artificially small to make examples easier to show.
          #column_names=CSV_COLUMNS, # supply column names if not given already in csv file
          #column_names=SELECT_COLUMNS #   you can specify few column names if required
          label_name=LABEL_COLUMN,
          na_value="?",
          num_epochs=1,
          ignore_errors=True,
          **kwargs)
    return dataset

def show_batch(dataset):
    print("------------------------------------")
    for batch, label in dataset.take(1): # read one batch and show it.
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

def show_packed_dataset(packed_dataset):
    for features, labels in packed_dataset.take(1):
        print(features.numpy())
        print()
        print(labels.numpy())

class PackNumericFeatures(object):
    #selects a list of numeric features and packs them into a single column
    def __init__(self, names):
        self.names = names
    def __call__(self, features, labels): # this allows to use class as function
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        return features, labels


def data_desc(file_path):
    desc = pd.read_csv(file_path)[NUMERIC_FEATURES].describe()
    print(desc)
    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])
    return MEAN,STD

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

def categorical_col():
    CATEGORIES = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
    }
    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    print(categorical_columns)
    return categorical_columns

def build_model(preprocessing_layer):
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    #print(model.summary())
    return model

def train_model(model,packed_train_data,packed_test_data):
    train_data = packed_train_data.shuffle(500)
    test_data = packed_test_data
    model.fit(train_data, epochs=20)
    test_loss, test_accuracy = model.evaluate(test_data)
    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
    return model

def predict_model(model,test_data):
    predictions = model.predict(test_data)

    # Show some results
    for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
        print("Predicted survival: {:.2%}".format(prediction[0]),
              " | Actual outcome: ",
              ("SURVIVED" if bool(survived) else "DIED"))




if __name__=="__main__":
    doConfig()
    train_file_path,test_file_path=getdatafiles()
    inspect_csv(train_file_path,test_file_path)
    inspect_pandas(train_file_path,test_file_path)
    raw_train_data=get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)
    show_batch(raw_train_data)                             # show five records

    SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
    DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
    temp_dataset = get_dataset(train_file_path,
                               select_columns=SELECT_COLUMNS,
                               column_defaults=DEFAULTS)
    packed_dataset = temp_dataset.map(pack)
    show_packed_dataset(packed_dataset)

    show_batch(temp_dataset)
    example_batch, labels_batch = next(iter(temp_dataset))
    print(example_batch)
    packed_ds=pack(example_batch,labels_batch)
    print(packed_ds)

    NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']
    packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
    packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
    show_batch(packed_train_data)
    example_batch, labels_batch = next(iter(packed_train_data))

    # numeric data setting
    train_mean,train_std=data_desc(train_file_path)
    test_mean, test_std = data_desc(test_file_path)
    normalizer = functools.partial(normalize_numeric_data, mean=train_mean, std=train_std)
    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer,shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]
    print(numeric_column)

    numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
    print(numeric_layer(example_batch).numpy())

    #categorical data settings
    categorical_columns=categorical_col()
    categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    print(categorical_layer(example_batch).numpy()[0])

    # create input layer that will extract and preprocess both input types
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
    print(preprocessing_layer(example_batch).numpy()[0])

    model=build_model(preprocessing_layer)
    model=train_model(model,packed_train_data,packed_test_data)
    predict_model(model, packed_test_data)



