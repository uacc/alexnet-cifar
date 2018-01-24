#!/usr/bin/env python
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import pdb
import math
import keras
from keras.datasets import cifar10
(x, y), (x_test, y_test) = cifar10.load_data()
Num_Cat = 10
print('Data Imported')
train_num = 1000
x = x[0:train_num]
y = y[0:train_num]

#Resize y as category form
import numpy as np
#Do we have to redefine y_train to fit the model?
y_train = np.zeros((y.size, Num_Cat))
i = 0
for j in y:
    y_train[i, j[0]] = 1
    i = i + 1

#Resize Image
import PIL
from PIL import Image

i = 0
x_train = np.zeros((x.shape[0],227,227,3))
for img in x:
    img = Image.fromarray(img.astype('uint8'),'RGB')
    im = img.resize((227,227),resample = PIL.Image.BILINEAR)
    tem = np.asarray(im).reshape((227,227,3))
    x_train[i] = tem
    if i%10000==0:
        print(i)
    i = i + 1
print('End of resize')
#pdb.set_trace()



#Building AlexNet
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D, BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras import optimizers
# import numpy as np  
seed = 7  
np.random.seed(seed)  
#learnrate = 0.01

for learnrate in np.arange(0.0091,0.012,0.001):
    print('Building model')
    model = Sequential()

    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    # Add normalization layer
    #model.add(BatchNormalization(axis = -1,momentum = 0.9,  epsilon = 0.01))

    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    #Add normalization layer
    #model.add(BatchNormalization(axis = -1, momentum = 0.9, epsilon = 0.01))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Flatten()) 
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(Num_Cat,activation='softmax'))

    #Setting optimizer with learning rate 0.01
    sgd = optimizers.SGD(lr = learnrate)
    #sgd = optimizers.Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)

    #===========Train Model===================
    print('Training Model')
    #Adding multi gpu

    #import tensorflow as tf
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #from keras.utils import multi_gpu_model
    #parallel_model = multi_gpu_model(model, gpus = 3)
    #parallel_model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
    model.compile(loss='categorical_crossentropy',optimizer=sgd) 

    #Adding check point to export weight from different epoch
    #from keras.callbacks import ModelCheckPoint
    #filepath = "weights-improvement-alexnet-{epoch:02d}.hsf5"
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss',  verbose = 1, save_best_only = True, mode = 'max', period = 10)

    #Fit model
    #history_callback = model.fit(x_train, y_train, epochs = 20, batch_size = 32, callbacks = [checkpoint])
    history_callback = model.fit(x_train, y_train, epochs = 20, batch_size = 32)
    #history_callback =parallel_model.fit(x_train, y_train, epochs = 6, batch_size = 16, callbacks = [checkpoint])
    losspath = "loss-"+str(learnrate)+".txt"
    loss_history  = history_callback.history["loss"]
    numpy_loss_history = np.array(loss_history)
    np.savetxt(losspath, numpy_loss_history, delimiter = ",")





















