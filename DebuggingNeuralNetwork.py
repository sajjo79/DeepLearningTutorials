from keras.layers import Input, BatchNormalization, Activation, Dropout
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
import cv2 as cv
from MyDataGenerator import MyImageDataConsumer
class DebuggingNeuralNetwork():
    def __init__(self):
        pass

    def visualize_weights_as_vals(self,layer):
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)  # (w,x,y,z) z number of filters of shape (w,x,y)
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        n_filters, ix = 6, 1  # plot first few filters
        np.set_printoptions(precision=3)
        ix = 1
        plt.figure(0)
        for i in range(n_filters):
            f = filters[:, :, :, i]
            # plot each channel separately
            for j in range(3):
                # specify subplot and turn of axis
                ax = plt.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.text(0.5, 0.5, f[:, :, j], horizontalalignment='center', verticalalignment='center')
                print(f[:, :, j])
                ix += 1
        plt.show()

    def visualize_weights_as_img(self,layer):
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)  # (w,x,y,z) z number of filters of shape (w,x,y)
        print('filter shape-->',filters.shape[0:3],"filters count-->",filters.shape[3])
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        n_filters, ix = 6, 1    #plot first few filters
        plt.figure(1)
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i] # i.e. [3,3,3,16] 16 filters each of shape 3x3x3
            # plot each channel separately
            for j in range(3):
                # specify subplot and turn of axis
                ax = plt.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(f[:, :, j], cmap='gray')
                print(f[:,:,j])
                ix += 1
        # show the figure
        #plt.show()

    def visualize_feature_maps(self,i,layer):
        print(i, layer.name, layer.output.shape)

    def plot_feature_maps(self,layer_name,feature_maps,f_map_size):
        print(feature_maps.shape)
        square = int(np.sqrt(f_map_size))
        ix = 1
        fig=plt.figure(0)
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                ix += 1
        fig.suptitle("Feature Maps:"+layer_name)
        plt.show()

    def list_layers(self,model):
        i=1
        for layer in model.layers:
            # check for convolutional layer
            print(i,layer.name,layer.output.shape)
            #if 'conv' not in layer.name:
            #    continue
            #self.visualize_weights_as_img(filters,biases)
            #self.visualize_weights_as_vals(filters, biases)
            #self.visualize_feature_maps(i,layer)
            #[7,15,23,31,39,48,57,66,76]
            i+=1
        print("--------------------------------------")

    def define_new_model_1(self,model):
        model = Model(inputs=model.inputs, outputs=model.layers[1].output)
        return model

    def test_new_model_1(self,model):
        img = cv.imread('bird.jpg') #target_size=(224, 224))
        img=cv.resize(img,(224,224))
        plt.imshow(img)
        plt.title("Original Image")
        img = np.expand_dims(img, axis=0)   # reshape to make it compatible with network input
        print(img.shape)
        feature_maps = model.predict(img)
        self.plot_feature_maps("first conv",feature_maps,feature_maps.shape[3])

    def define_new_model_2(self,model):
        #return model
        #ixs = [2, 5, 9, 13, 17]
        print(len(model.layers))
        outputs = [model.layers[i].output for i in range(len(model.layers))]
        model = Model(inputs=model.inputs, outputs=outputs)
        return model

    def test_new_model_2(self,model):
        layers=[layer.name for layer in model.layers]
        print(layers)
        img = cv.imread('bird.jpg')  # target_size=(224, 224))
        img = cv.resize(img, (224, 224))
        plt.imshow(img)
        plt.title("Original Image")
        img = np.expand_dims(img, axis=0)  # reshape to make it compatible with network input
        print(img.shape)
        feature_maps = model.predict(img)
        i=0
        for fm in feature_maps:
            print(fm.shape)
            self.plot_feature_maps(layers[i]+str(fm.shape),fm, fm.shape[3])
            i=i+1

if __name__=="__main__":
    train_images_dir = 'E:\\PyCharmProjects\\TF_tutorials\\tgs-salt-identification-challenge\\train\\imgs\\images\\*'
    train_masks_dir = 'E:\\PyCharmProjects\\TF_tutorials\\tgs-salt-identification-challenge\\train\\msks\\masks\\*'
    midc = MyImageDataConsumer()
    data_gen = midc.get_data(train_images_dir, train_masks_dir)

    input_img = Input((224, 224, 3), name='img')
    model = midc.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    dnn=DebuggingNeuralNetwork()
    dnn.list_layers(model)

    # model_1=dnn.define_new_model_1(model)
    # dnn.list_layers(model_1)
    # dnn.test_new_model_1(model_1)

    model_2 = dnn.define_new_model_2(model)
    dnn.list_layers(model_2)
    dnn.test_new_model_2(model_2)

    #results = midc.train_model(model, data_gen)
    #midc.plot_loss(results)

