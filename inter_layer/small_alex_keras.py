#!/usr/bin/env python
import pdb
import os
import math
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import math
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D, BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint

#==========Experiment parameter==========
SAVEMARGIN = False 
SAVEWEIGHT = False 
SAVEOUTPUT = False
EXCESSRISK = False 
LEARNRATE = 0.01
DECAYFACTOR = 0.95
MOMENTUM = 0.9
step = 0.00005
EPOCHS = 20 
PERIOD = 5 
BATCHSIZE = 16
TRAIN_NUM = 50000
FIRST_LAYER_DEPTH = 256 
SECOND_LAYER_DEPTH = 384 

def importdata():
    from keras.datasets import cifar10
    (x, y), (x_test, y_test) = cifar10.load_data()
    Num_Cat = 10
    print('Data Imported')
    x = x[0:TRAIN_NUM]
    y = y[0:TRAIN_NUM]
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


def BuildModel(firstlayerdepth, secondlayerdepth):
    model = Sequential()
    model.add(Conv2D(firstlayerdepth,(5, 5),strides=(1,1),input_shape=(28, 28, 3),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
    model.add(BatchNormalization())

    # LRN layer 

    model.add(Conv2D(secondlayerdepth,(5, 5),strides=(1,1), padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
    model.add(BatchNormalization())

    #LRN layer

    # Dense layer add dropout
    model.add(Flatten()) 
    model.add(Dense(384, activation = 'relu'))
    model.add(Dense(192, activation = 'relu'))
    model.add(Dense(10,activation='softmax'))  
    return model

## Adding  new model to get the output layer
#from keras.models import Model
#layer_name = 'dense_1'
#intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)


#===========Train Model===================
def trainmodel(FIRST_LAYER_DEPTH, SECOND_LAYER_DEPTH):
    x_train, y_train, x_test, y_test = importdata()

    model = BuildModel(FIRST_LAYER_DEPTH, SECOND_LAYER_DEPTH)
    print "first layer: " + str(FIRST_LAYER_DEPTH)
    print "second layer: " + str(SECOND_LAYER_DEPTH)

    sgd = optimizers.SGD(lr = LEARNRATE, decay = DECAYFACTOR, momentum = MOMENTUM) 
    model.compile(loss='categorical_crossentropy',optimizers=sgd, metrics=['accuracy'])
    model.summary()

    #Adding check point to export weight from different epoch
    filepath = "noise-weights-improvement-"+str(FIRST_LAYER_DEPTH) + str(SECOND_LAYER_DEPTH) +"-conn-{epoch:02d}.hsf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss',  verbose = 1, save_best_only = False, mode = 'min', period = PERIOD)
    #Import callback function to compute margin
    import my_callbacks
    marginhistory = my_callbacks.Histories()

           #========== Fit model==========
    if EXCESSRISK == True:
        print "Saving excessrisk for each epoch"
        #history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_test, y_test), callbacks = [checkpoint, marginhistory] )
        history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_test, y_test))
        acc_history = history_callback.history["acc"]
        val_acc_history = history_callback.history["val_acc"]
        numpy_acc_history = np.array(acc_history)
        numpy_val_acc_history = np.array(val_acc_history)
        accpath = "noise-acc_history_conn_net" + str(FIRST_LAYER_DEPTH) + str(SECOND_LAYER_DEPTH) + ".txt"
        valpath = "noise-val_acc_history_conn_net" + str(FIRST_LAYER_DEPTH) + str(SECOND_LAYER_DEPTH)+ ".txt"
        np.savetxt(accpath,numpy_acc_history, delimiter = ",")
        np.savetxt(valpath,numpy_val_acc_history, delimiter = ",")
#else:
#    if SAVEMARGIN == True and SAVEWEIGHT == True:
#        print "Saving margin and weight for each epoch"
#        history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_train, y_train), callbacks = [checkpoint, marginhistory] )
#    elif SAVEMARGIN == True:
#        print "Saving margin for each epoch"
#        history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_train, y_train), callbacks = [marginhistory] )
#    elif SAVEWEIGHT == True:
#        print "Saving weight for each epoch"
#        history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_train, y_train), callbacks = [checkpoint] )

# Print out margins after each epoch
#if SAVEOUTPUT == True:
#    print "Saving output to compute margin distribution"
#    print(marginhistory.margins)
#    margin_history = marginhistory.margins
#    margin_his = np.array(margin_history)
#    np.savetxt('noise-margin_history_train_margin.txt', margin_his, delimiter = ",")
#
#    intermediate_output = intermediate_layer_model.predict(x_train)
#    final_output = np.array(intermediate_output)
#    np.savetxt('noise-final_output.txt', final_output, delimiter = ",")
#    np.savetxt('noise-y_train.txt', y_train, delimiter = ",")

