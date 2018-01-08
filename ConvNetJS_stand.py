#!/usr/bin/env python
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import pdb
import math
import keras
#Import data
# import numpy as np
# x_train = np.load('x_train.npy')
# y_train = np.load('y_train.npy')
# print('Load Data')
from keras.datasets import cifar10
(x, y), (x_test, y_test) = cifar10.load_data()
Num_Cat = 10
print('Data Imported')
train_num = 1000
x = x[0:train_num]
y = y[0:train_num]

# #===============Resize y as category form==============
# import numpy as np
# #Do we have to redefine y_train to fit the model?
# y_train = np.zeros((y.size, Num_Cat))
# i = 0
# for j in y:
#     y_train[i, j[0]] = 1
#     i = i + 1

# #Resize Image
# import PIL
# from PIL import Image

# i = 0
# x_train = np.zeros((x.shape[0],227,227,3))
# for img in x:
#     img = Image.fromarray(img.astype('uint8'),'RGB')
#     im = img.resize((227,227),resample = PIL.Image.BILINEAR)
#     tem = np.asarray(im).reshape((227,227,3))
#     x_train[i] = tem
#     if i%10000==0:
#         print(i)
#     i = i + 1
# print('End of resize')
# #pdb.set_trace()


#Calculate the l2 norm of the x_train
#from numpy import linalg as LA
#(n, x, y, z) = x_train.shape
#tem = 0
#for i in range(0,n):
#    img = x_train[i,:,:,:]
#    img = img.reshape((x*y*z, 1))
#    try:
#        tem = tem + math.pow(LA.norm(img, 'fro'), 2)
#    except:
#        pdb.set_trace()
#            
#print(math.sqrt(tem))
#

#Building ConvNetJS
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras import optimizers
# import numpy as np  
seed = 7  
np.random.seed(seed)  
  
print('Building model')
model = Sequential()

model.add(ZeroPadding2D(padding=((3,2),(3,2)) )
model.add(Conv2D(16,(5,5),strides=(1,1),input_shape=(32,32,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(ZeroPadding2D(padding=((3,2),(3,2)))
model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(ZeroPadding2D(padding=((3,2),(3,2)))
model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# model.add(ZeroPadding2D(padding=(1,1)))
# model.add(Conv2D(384,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
# model.add(ZeroPadding2D(padding=(1,1)))
# model.add(Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Flatten()) 
model.add(Dense(320,activation='softmax'))  
# model.add(Dropout(0.5))  
# model.add(Dense(4096,activation='relu'))  
# model.add(Dropout(0.5))  
# model.add(Dense(Num_Cat,activation='softmax'))

#Setting optimizer with learning rate 0.01
sgd = optimizers.SGD(lr = 0.005)

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
# filepath = "weights-improvement-{epoch:02d}.hsf5"
# checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss',  verbose = 1, save_best_only = True, mode = 'max', period = 2)

# #Fit model
# history_callback = model.fit(x_train, y_train, epochs = 6, batch_size = 16, callbacks = [checkpoint])
# #history_callback =parallel_model.fit(x_train, y_train, epochs = 6, batch_size = 16, callbacks = [checkpoint])
# loss_history  = history_callback.history["loss"]
# numpy_loss_history = np.array(loss_history)
# np.savetxt("loss_history.txt", numpy_loss_history, delimiter = ",")

# pdb.set_trace()


















