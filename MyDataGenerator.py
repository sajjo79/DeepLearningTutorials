import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import glob


class MyImageDataGenerator():
    def __init__(self,rescale):
        self.rescale=rescale
        self.IMG_HEIGHT=128
        self.IMG_WIDTH=128

    def __iter__(self):
        return self

    def preprocess_image(self,image):
        image=image/(1./255)
        image = cv.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH))
        '''
        --- Rescale Image
        --- Rotate  Image
        --- Resize  Image
        --- Flip    Image
        --- PCA    etc.
        '''
        return (image)

    def read_images(self,batch_files):
        images=[]
        for file in batch_files:
            img = cv.imread(file)
            img=self.preprocess_image(img)
            images+=[img]
        return images

    def flow_from_directory(self, batch_size, directory, shuffle, seed):
        data_files = glob.glob(directory)
        fcount=len(data_files)
        if shuffle==True:
            np.random.seed(seed)
            np.random.shuffle(data_files)
        batch_start=0
        while True:
            batch_files=data_files[batch_start:batch_start+batch_size]
            batch_images=self.read_images(batch_files)
            batch_imgs = np.array(batch_images)
            if batch_start+batch_size==fcount: batch_start=0
            else: batch_start+=batch_size
            yield (batch_imgs)

#=============================================================================================
class MyImageDataConsumer():
    def __init__(self):
        pass
    def get_data(self,train_image_dir,train_mask_dir):
            image_generator = MyImageDataGenerator(rescale=1. / 255)  # Generator for our training data
            img_gen = image_generator.flow_from_directory(batch_size=20,
                                                                  directory=train_image_dir,
                                                                  shuffle=True,
                                                                  seed=1)

            mask_generator = MyImageDataGenerator(rescale=1. / 255)  # Generator for our training data
            msk_gen = mask_generator.flow_from_directory(batch_size=20,
                                                          directory=train_mask_dir,
                                                          shuffle=True,
                                                          seed=1)
            data_gen=zip(img_gen,msk_gen)
            return data_gen
#=============================================================================================
# Network
    def conv2d_block(self,input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def get_unet(self,input_img, n_filters=16, dropout=0.5, batchnorm=True):
        # contracting path
        c1 = self.conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

        # expansive path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

        outputs = Conv2D(3, (1, 1), activation='sigmoid')(c9) # 1 --> 3
        model = Model(inputs=[input_img], outputs=[outputs])

        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        print(model.summary())
        tf.keras.utils.plot_model(model, '../saltseg.png', show_shapes=True)

        return model
#=============================================================================================
    def train_model(self,model,train_data_gen):
        callbacks = [
            EarlyStopping(patience=10, verbose=1), # Number of epochs with no improvement after which training will be stopped.
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]

        results=model.fit_generator(
            train_data_gen,
            steps_per_epoch=200,
            epochs=5,
            callbacks=callbacks)


        # results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
        #                     validation_data=(X_valid, y_valid))
        return results
#=============================================================================================
    def plot_loss(self,results):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.title("Learning curve")
        plt.plot(results.history["loss"], label="loss")
        #plt.plot(results.history["val_loss"], label="val_loss")
        #plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()
        plt.show()

if __name__=="__main__":
    train_images_dir = 'E:\\PyCharmProjects\\TF_tutorials\\tgs-salt-identification-challenge\\train\\imgs\\images\\*'
    train_masks_dir = 'E:\\PyCharmProjects\\TF_tutorials\\tgs-salt-identification-challenge\\train\\msks\\masks\\*'
    midc=MyImageDataConsumer()
    data_gen=midc.get_data(train_images_dir,train_masks_dir)

    input_img = Input((128, 128,3), name='img')
    model = midc.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    results=midc.train_model(model,data_gen)
    midc.plot_loss(results)

