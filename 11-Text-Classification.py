from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# ----- Globals --------------------------------
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

def get_data():
    for name in FILE_NAMES:
        text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name, cache_dir="E:\\PyCharmProjects\\TF_tutorials\\Datasets\\")
    parent_dir = os.path.dirname(text_dir)
    print(parent_dir)
    return parent_dir

def labeler(example, index):
  return example, tf.cast(index, tf.int64)

def load_text_to_datasets(parent_dir):
    labeled_data_sets = []
    for i, file_name in enumerate(FILE_NAMES):
      lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
      labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
      labeled_data_sets.append(labeled_dataset)

    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    for ex,lbl in all_labeled_data.take(5):
        print(ex.numpy(),lbl.numpy())

    return all_labeled_data

def build_vocabulary(all_labeled_data):
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    print(vocab_size)
    return vocabulary_set,vocab_size

def encode_examples(vocabulary_set):
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    example_text = next(iter(all_labeled_data))[0].numpy()
    print(example_text)
    encoded_example = encoder.encode(example_text)
    print(encoded_example)
    return encoder

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  #Wraps a python function into a TensorFlow op that executes it eagerly
  # func: A Python function which accepts a list of Tensor objects
  # inp: A list of Tensor objects.
  #Tout: A list or tuple of tensorflow data types indicating what function returns
  return tf.py_function(func=encode, inp=[text, label], Tout=(tf.int64, tf.int64))

def train_test_split(all_encoded_data,vocab_size):
    # test_data and train_data are not collections of (example, label) pairs, but collections of batches.
    # Each batch is a pair of (many examples, many labels) represented as arrays.
    train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    test_data = all_encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    sample_text, sample_labels = next(iter(test_data))
    print(sample_text[0], sample_labels[0])
    vocab_size += 1
    return train_data,test_data,vocab_size

def build_n_compile_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    for units in [64, 64]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_n_evaluate(model,train_data,test_data):
    model.fit(train_data, epochs=3, validation_data=test_data)
    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
    return model

if __name__=="__main__":
    parent_dir=get_data()
    all_labeled_data=load_text_to_datasets(parent_dir)
    vocabulary_set,vocab_size=build_vocabulary(all_labeled_data)
    encoder=encode_examples(vocabulary_set)

    all_encoded_data = all_labeled_data.map(encode_map_fn)
    train_data,test_data,vocab_size=train_test_split(all_encoded_data,vocab_size)
    model=build_n_compile_model()
    train_n_evaluate(model,train_data,test_data)
