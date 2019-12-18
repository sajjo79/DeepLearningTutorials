from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf

def getdata():
    csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
    df = pd.read_csv(csv_file)
    return df #data frame

def inspect_data(df):
    print(df.head())
    print(df.dtypes)
    df['thal'] = pd.Categorical(df['thal']) # Convert thal column which is an object in the dataframe to a discrete numerical value.
    df['thal'] = df.thal.cat.codes
    print(df.head())
    print(df.dtypes)

def loading_data(df):
    #Load data using tf.data.Dataset
    target = df.pop('target')   # remove target column
    print(df.head)
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    thal=tf.constant(df['thal'])
    train_DS = dataset.shuffle(len(df))
    train_DS=train_DS.batch(1)
    return train_DS,thal,target

def view_data(df,target):
    dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
    for dict_slice in dict_slices.take(1):
        print(dict_slice)
    return dict_slices


def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

def alternate_model(df):
    # using keras functional model
    inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
    x = tf.stack(list(inputs.values()), axis=-1)

    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model= tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    return model

def alternate_training(model,dict_slices):
    model.fit(dict_slices, epochs=15)

def train_model(model,train_DS):
    model.fit(train_DS, epochs=15)

if __name__=="__main__":
    df=getdata()
    inspect_data(df)
    train_DS,thal,target=loading_data(df)
    model=get_compiled_model()
    model=train_model(model, train_DS)
    dict_slices=view_data(df,target)
    model=alternate_model(df)
    alternate_training(model,dict_slices)