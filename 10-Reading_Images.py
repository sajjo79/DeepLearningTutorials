# This tutorial provides a simple example of how to load an image dataset using tf.data.
# The dataset used in this example is distributed as directories of images, with one class of image per directory.
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time

"""
The number of elements to prefetch should be equal to (or possibly greater than) the number of batches consumed by 
a single training step. You could either manually tune this value, or set it to tf.data.experimental.AUTOTUNE 
which will prompt the tf.data runtime to tune the value dynamically at runtime.
"""
AUTOTUNE = tf.data.experimental.AUTOTUNE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pathlib
import matplotlib.pyplot as plt
import sys
# = Globals variables and constants =====================
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(3670 / BATCH_SIZE)
CLASS_NAMES=""
default_timeit_steps = 1000

def getData():
    #data_dir = tf.keras.utils.get_file(
    #    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    #    fname='flower_photos', untar=True)
    #data_dir = pathlib.Path(data_dir)
    #print(data_dir)
    #data_dir="C:\Users\pc\.keras\datasets\flower_photos"
    data_dir="E:\\PyCharmProjects\\TF_tutorials\\Datasets\\flower_photos"
    data_dir = Path(data_dir)
    return data_dir

def inspectData(data_dir):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    print(CLASS_NAMES)
    roses = list(data_dir.glob('roses/*'))
    for image_path in roses[:3]: # display three images
        plt.imshow(Image.open(str(image_path)))
        plt.show()
    return image_count, CLASS_NAMES

def loadImages(data_dir,image_count,CLASS_NAMES):
    """
    tf.keras.preprocessing is Keras data preprocessing utils that can process image, text and sequential data
    class ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation.
    API details can be seen at
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?version=stable
    A simple way to load images is to use tf.keras.preprocessing.
    The 1./255 is to convert from uint8 to float32 in range [0,1].
    The following keras.preprocessing method is convienient, but has two downsides:
        It's slow. See the performance section below.
        It lacks fine-grained control.
        It is not well integrated with the rest of TensorFlow.

    """
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH), # resize all images to same size
                                                         classes=list(CLASS_NAMES))
    return train_data_gen

def showBatch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()

def loadAsDataset(data_dir):
    """
    A dataset contains elements that each have the same (nested) structure and
    the individual components of the structure can be of any type representable by tf.TypeSpec,
    """
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
    for f in list_ds.take(5):
        print(f.numpy())
    return list_ds

# Write a short pure-tensorflow function that converts a file paths to an (image_data, label) pair:
def get_label(file_path,debug=0):
  parts = tf.strings.split(file_path, os.path.sep)  #  convert the path to a list of path components
  if debug==1:
      print("file path -->",file_path)
      print("parts -->",parts)                          # Bytes literals are always prefixed with 'b' or 'B'
      print(parts[-2] == CLASS_NAMES)
  return parts[-2] == CLASS_NAMES                   # The second to last is the class-directory

def decode_img(img,debug=0):
  img1 = tf.image.decode_jpeg(img, channels=3)          # Decode a JPEG-encoded image to a uint8 tensor.
  img2 = tf.image.convert_image_dtype(img1, tf.float32) # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img3=tf.image.resize(img2, [IMG_WIDTH, IMG_HEIGHT])  # resize the image to the desired size.
  if debug==1:
      print(sys.getsizeof(img),type(img),"\n",
            sys.getsizeof(img1),type(img),"\n",
            sys.getsizeof(img2),type(img),"\n",
            sys.getsizeof(img3),type(img),"\n"
            )
      im=img.numpy()
      im1=img1.numpy()
      im2=img2.numpy()
      im3=img3.numpy()
      print(sys.getsizeof(im),type(im),"\n",
            sys.getsizeof(im1),type(im1),"\n",
            sys.getsizeof(im2),type(im2),"\n",
            sys.getsizeof(im3),type(im3),"\n",)
  return img3

def process_path(file_path):
  print(file_path)
  label = get_label(file_path)
  img = tf.io.read_file(file_path)                  # load the raw data from the file as a string
  img = decode_img(img)
  return img, label

def print_data(list_ds):
    test=list_ds.take(1)
    for f in test:
        get_label(f.numpy(),debug=1)
        img = tf.io.read_file(f.numpy())
        decode_img(img,debug=1)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())
    return labeled_ds

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use '.cache(filename)' to cache preprocessing work for datasets that don't fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.repeat()              # Repeat forever
  ds = ds.batch(BATCH_SIZE)
  # 'prefetch' lets the dataset fetch batches in the background while the model is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

if __name__=="__main__":
    data_dir=getData()
    image_count,CLASS_NAMES=inspectData(data_dir)
    train_data_gen=loadImages(data_dir,image_count,CLASS_NAMES)
    image_batch, label_batch = next(train_data_gen)
    showBatch(image_batch, label_batch)
    list_ds=loadAsDataset(data_dir)
    labeled_ds=print_data(list_ds)
    train_ds=prepare_for_training(labeled_ds)
    image_batch, label_batch = next(iter(train_ds))
    showBatch(image_batch.numpy(), label_batch.numpy())

    import gc
    gc.collect()
    gc.collect()
    gc.collect()
    #Performance Comparisions
    # 'keras.preprocessing
    timeit(train_data_gen)  # 1000 batches: 65.76107859611511 s  486.61002 Images/s

    # `tf.data`
    timeit(train_ds)        # 1000 batches: 13.73485016822815 s  2329.83976 Images/s

    uncached_ds = prepare_for_training(labeled_ds, cache=False)
    timeit(uncached_ds)     # 1000 batches: 52.65759873390198 s  0607.69957 Images/s

    filecache_ds = prepare_for_training(labeled_ds, cache="./flowers.tfcache")
    timeit(filecache_ds)    # 1000 batches: 89.83073210716248 s 356.22553 Images/s

    # So tf.data is very fast in our experiment
