#!/usr/bin/env python
import pdb
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import pdb
import math
import keras
#Import data
import numpy as np
# x_train = np.load('x_train.npy')
# y_train = np.load('y_train.npy')
# print('Load Data')
from keras.datasets import cifar10
(x, y), (x_test, y_test) = cifar10.load_data()
Num_Cat = 10
print('Data Imported')
train_num = 50000
x = x[0:train_num]
y = y[0:train_num]

#===============Resize y as category form==============
import numpy as np
#Do we have to redefine y_train to fit the model?
y_train = np.zeros((y.size, Num_Cat))
i = 0
for j in y:
    y_train[i, j[0]] = 1
    i = i + 1

#padding zeros to first layer
shape_x = x.shape
tem = np.zeros((shape_x[0],shape_x[1]+5, shape_x[2]+5, shape_x[3]))
tem[:, 3:3+shape_x[1],3:3+shape_x[2], :] = x
x_train = tem
#pdb.set_trace()

#Building ConvNetJS
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras import optimizers
import numpy as np  
seed = 7  
np.random.seed(seed)  
  
for learnrate in np.arange(0.0001, 0.005,0.001):
    print('Building model')
    model = Sequential()

    #model.add(ZeroPadding2D(padding=((3,2),(3,2))))
    model.add(Conv2D(16,(5,5),strides=(1,1),input_shape=(37,37,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(ZeroPadding2D(padding=((3,2),(3,2))))
    model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(ZeroPadding2D(padding=((3,2),(3,2))))
    model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


    model.add(Flatten()) 
    model.add(Dense(10,activation='softmax'))  

    #Setting optimizer with learning rate 0.01
    sgd = optimizers.SGD(lr = learnrate)
    #sgd = optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)

    #===========Train Model===================
    print('Training Model')
    model.compile(loss='categorical_crossentropy',optimizer=sgd) 

    #Adding check point to export weight from different epoch
    from keras.callbacks import ModelCheckpoint
    filepath = "weights-improvement-"+str(learnrate)+"conn-{epoch:02d}.hsf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss',  verbose = 1, save_best_only = True, mode = 'max', period = 10)

    # #Fit model
    history_callback = model.fit(x_train, y_train, epochs = 20, batch_size =16, callbacks = [checkpoint])
    loss_history  = history_callback.history["loss"]
    numpy_loss_history = np.array(loss_history)
    losspath = "loss_history_conn_net"+str(learnrate)
    np.savetxt(losspath, numpy_loss_history, delimiter = ",")




















