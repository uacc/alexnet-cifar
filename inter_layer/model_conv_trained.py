from __future__ import print_function
import pdb
from glob import glob
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import math
import tensorflow as tf
import numpy as np
from sklearn.svm import LinearSVC

#from small_alex_keras import BuildModel, importdata
import keras
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D, BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from keras.datasets import cifar10
from data import get_mnist_data, num_classes, input_shape
from read_activations import get_activations, display_activations
CONV_MODEL = True
SMALL_ALEX = True
VGG_19 = True
NUM_CLASSES = 10
LEARNRATE = 0.01
DECAYFACTOR = 0.95
MOMENTUM = 0.9

def importdata():
    (x, y), (x_test, y_test) = cifar10.load_data()
    Num_Cat = 10
    print('Data Imported')
    #==========input data preprocessing ==========
    # Rescale to [0, 1]
    x_train = x / 255.
    x_test = x_test / 255.
    # Crop from center to 28*28
    x_train = x_train[:, 2:30, 2:30, :]
    x_test = x_test[:, 2:30, 2:30, :]
    # tf.per_image_standardization 
    train_mean = np.mean(x_train, axis = (1, 2, 3))
    test_mean = np.mean(x_test, axis = (1, 2, 3))
    train_std = np.std(x_train, axis = (1, 2, 3))
    test_std = np.std(x_test, axis = (1, 2, 3))
    adjusted_value = 1./math.sqrt(28*28*3)
    for i in range(len(train_mean)):
        x_train[i, :, :, :] = x_train[i, :, :, :] - train_mean[i]
        adjusted_std = np.max((train_std[i], adjusted_value))
        x_train[i, :, :, :] = x_train[i, :, :, :] / adjusted_std 

    for i in range(len(test_mean)):
        x_test[i, :, :, :] = x_test[i, :, :, :] - test_mean[i]
        adjusted_std = np.max((test_std[i], adjusted_value))
        x_test[i, :, :, :] = x_test[i, :, :, :] / adjusted_std 

    temtrain = np.zeros((y.size, Num_Cat))
    temtest = np.zeros((y_test.size, Num_Cat))
    i = 0
    for j in y:
        temtrain[i, j[0]] = 1
        i = i + 1
    i = 0
    for j in y_test:
        temtest[i, j[0]] = 1
        i = i + 1
    y_train = temtrain
    y_test = temtest

    return x_train, y_train, x_test, y_test


def conv_model():
    model = Sequential()

    model.add(Conv2D(16,(5,5),strides=(1,1),input_shape=(36,36,3),padding='valid',activation='relu',kernel_initializer='uniform', name = 'conv_1'))  
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name = 'pool_1'))

    model.add(ZeroPadding2D(padding=((2,2),(2,2))))
    model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform', name = 'conv_2'))  
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name = 'pool_2'))

    model.add(ZeroPadding2D(padding=((2,2),(2,2))))
    model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform', name = 'conv_3'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name = 'pool_3'))


    model.add(Flatten(name = 'flat')) 
    model.add(Dense(10,activation='softmax', name = 'fc_1'))  
    return model

def small_alex():
    model = Sequential()
    model.add(Conv2D(firstlayerdepth,(5, 5),strides=(1,1),input_shape=(28, 28, 3),padding='valid',activation='relu',kernel_initializer='uniform', name = 'conv_1'))  
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), name = 'pool_1'))
    model.add(BatchNormalization(name = 'norm_1'))

    model.add(Conv2D(secondlayerdepth,(5, 5),strides=(1,1), padding='valid',activation='relu',kernel_initializer='uniform', name = 'conv_2'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), name = 'pool_2'))
    model.add(BatchNormalization(name = 'norm_2'))

    # Dense layer without add dropout
    model.add(Flatten()) 
    model.add(Dense(384, activation = 'relu', name = 'fc_1'))
    model.add(Dense(192, activation = 'relu', name = 'fc_2'))
    return model

def vgg_19():
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (32, 32, 3), name = 'block1_conv1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding = 'same', name = 'block1_conv2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')) 

    model.add(Conv2D(128, (3, 3), padding = 'same', name = 'block2_conv1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding = 'same', name = 'block2_conv2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')) 

    
    model.add(Conv2D(256, (3, 3), padding = 'same', name = 'block3_conv1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding = 'same', name = 'block3_conv2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding = 'same', name = 'block3_conv3'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding = 'same', name = 'block3_conv4'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')) 


    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block4_conv1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block4_conv2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block4_conv3'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block4_conv4'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')) 

    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block5_conv1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block5_conv2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block5_conv3'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding = 'same', name = 'block5_conv4'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))

    model.add(Flatten())
   
    model.add(Dense(4096, kernel_regularizer = regularizers.l2(WEIGHTDECAY), name = 'fc1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, kernel_regularizer = regularizers.l2(WEIGHTDECAY), name = 'fc2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, name = 'fc3'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('softmax'))
    
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_interlayer(MODEL_NAME = 'conv_model', LAYER_NAME = 'fc_1'):
    if MODEL_NAME == 'small_alex':
        x_train, y_train, x_test, y_test = importdata()
        model = small_alex()
        model.load_weights('small_alex_weight.hsf5')
    elif MODEL_NAME == 'vgg_19':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        model = vgg_19()
        model.load_weights('vgg_19_weight.hsf5')
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        shape_x = x_train.shape
        shape_test = x_test.shape
        tem = np.zeros((shape_x[0],shape_x[1]+4, shape_x[2]+4, shape_x[3]))
        tem_test = np.zeros((shape_test[0],shape_test[1]+4, shape_test[2]+4, shape_test[3]))
        tem[:, 2:2+shape_test[1],2:2+shape_test[2], :] = x_train
        tem_test[:, 2:2+shape_test[1],2:2+shape_test[2], :] = x_test
        x_train = tem
        x_test = tem_test

        model = conv_model()
        #model.load_weights('conv_weight.hsf5')
        model.load_weights('noise_conv_weight.hsf5')
        
    sgd = optimizers.SGD(lr = LEARNRATE, decay = DECAYFACTOR, momentum = MOMENTUM) 
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    model.summary()
    
    test_inter_layer = get_activations(model, x_test, print_shape_only = True, layer_name = LAYER_NAME)
    train_inter_layer = get_activations(model, x_train, print_shape_only = True, layer_name = LAYER_NAME)
    inter_layer = np.array(test_inter_layer)
    train_inter_layer = np.array(train_inter_layer)
    return inter_layer[0], train_inter_layer[0]

def classification_acc(inter_layer, train_inter_layer):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Linear SVC classifier
    y_test = y_test.reshape((10000,))
    #Modify inter_layer train_inter_layer shape
    inter_layer = inter_layer.reshape((inter_layer.shape[0], inter_layer.shape[1] * inter_layer.shape[2] * inter_layer.shape[3]))
    train_inter_layer = train_inter_layer.reshape((train_inter_layer.shape[0], train_inter_layer.shape[1] * train_inter_layer.shape[2] * train_inter_layer.shape[3]))
    clf = LinearSVC(random_state=0)
    #===================Train SVC==========
    print("Training SVC")
    clf.fit(inter_layer, y_test)
    result = clf.predict(inter_layer) - y_test
    acc = np.count_nonzero(result) * 1.
    test_acc = acc / y_test.shape[0]
    
    #Linear SVC on training
    y_train = y_train.reshape((50000, ))
    clf.fit(train_inter_layer, y_train)
    result = clf.predict(train_inter_layer) - y_train
    acc = np.count_nonzero(result) * 1.
    train_acc = acc / y_train.shape[0]

    return test_acc, train_acc
    


inter_layer, train_inter_layer = get_interlayer('conv_model', 'pool_1' )
test_1 = classification_acc(inter_layer, train_inter_layer)
inter_layer, train_inter_layer = get_interlayer('conv_model', 'pool_2' )
test_2 = classification_acc(inter_layer, train_inter_layer)
#inter_layer, train_inter_layer = get_interlayer('conv_model', 'pool_3' )
#test_3 = classification_acc(inter_layer, train_inter_layer)
pdb.set_trace()

