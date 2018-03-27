#!/usr/bin/env python
import pdb
import os
import math
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import math
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D, Activation
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

SAVEMARGIN = False
SAVEWEIGHT = False
SAVEOUTPUT = False
EXCESSRISK = False 
LEARNRATE = 0.05
WEIGHTDECAY = 0.0005 
MOMENTUM = 0.9
EPOCHS = 300 
PERIOD = EPOCHS - 2 
BATCHSIZE = 128 
NUM_CLASSES = 10
BATCH_NORM = False
PRETRAIN = False

#==========Learning rate decay every 30 epochs==========
#==========Using callback functions=========
class decay_lr(keras.callbacks.Callback):
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch = n_epoch
        self.decay = decay

    def on_epoch_begin(self, epoch, logs = {}):
        if epoch > 1 and epoch%self.n_epoch == 0:
            optimizer = self.model.optimizer
            lr = K.eval(optimizer.lr * decay)
            K.set_value(optimizer.lr, lr)


#==========Import data==========
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np.random.randint(0,10,size = len(y_train))
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#==========Building Model==========
def VGG19(weights_path = None):
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

if PRETRAIN:
    model = VGG19('vgg19_weights.h5')
else:
    model = VGG19()
model.summary()

sgd = optimizers.SGD(lr = LEARNRATE,  momentum = MOMENTUM)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint
filepath = "vgg-noise-weights-improvement-conn-{epoch:02d}.hsf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss',  verbose = 1, save_best_only = False, mode = 'min', period = PERIOD)
#decaySchedule = decay_lr(30, 0.5)
#history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_test, y_test), callbacks = [decaySchedule, checkpoint])
history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_test, y_test), callbacks = [checkpoint])
acc_history = history_callback.history["acc"]
val_acc_history = history_callback.history["val_acc"]
numpy_acc_history = np.array(acc_history)
numpy_val_acc_history = np.array(val_acc_history)
accpath = "acc_vgg19.txt"
valpath = "val_acc_vgg19.txt"
np.savetxt(accpath,numpy_acc_history, delimiter = ",")
np.savetxt(valpath,numpy_val_acc_history, delimiter = ",")
