#!/usr/bin/env python
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import math
import keras
import numpy as np
import tensorflow as tf

#==========Experiment parameter==========
LEARNRATEFINDING = False
SAVEMARGIN = False 
SAVEWEIGHT = False 
SAVEOUTPUT = False
EXCESSRISK = False 
#MARGIN = False
lowerbound = 0.0001
upperbound = 0.0006
learnrate = 0.0001
step = 0.00005
EPOCHS = 200 
PERIOD = 5 
BATCHSIZE = 16

#==========Import Data==========
from keras.datasets import cifar10
(x, y), (x_test, y_test) = cifar10.load_data()
Num_Cat = 10
print('Data Imported')
train_num = 50000
x = x[0:train_num]
y = y[0:train_num]

#==========Normalize the training data==========

#==========Building ConvNetJS==========
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras import optimizers
import numpy as np  
seed = 7  
np.random.seed(seed)  
  
#for learnrate in np.arange(0.0001, 0.005,0.001):
print('Building model')
model = Sequential()

#model.add(ZeroPadding2D(padding=((3,2),(3,2))))
model.add(Conv2D(16,(5,5),strides=(1,1),input_shape=(36,36,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(ZeroPadding2D(padding=((2,2),(2,2))))
model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(ZeroPadding2D(padding=((2,2),(2,2))))
model.add(Conv2D(20,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


model.add(Flatten()) 
model.add(Dense(10,activation='softmax'))  

# Adding  new model to get the output layer
from keras.models import Model
layer_name = 'dense_1'
intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)


#===========Train Model===================
print('Training Model')
#Setting optimizer with learning rate 0.01
if LEARNRATEFINDING == True:
    for learnrate in np.arange(lowerbound, upperbound, step):
        #========== Randomize weights in each loop ==========
        if learnrate != lowerbound:
            weights = model.get_weights()
            weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
            model.set_weights(weights)
        sgd = optimizers.SGD(lr = learnrate)
        #sgd = optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)

        model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
        #model.summary()
        #model.fit(x_train, y_train, epochs = EPOCHS, validation_data = (x_test, y_test), batch_size =BATCHSIZE)
        print "Start with learning rate:"
        print learnrate
        model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCHSIZE)
else:
    sgd = optimizers.SGD(lr = learnrate)
    print "Using fixed learning rate " + str(learnrate) 
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    #model.summary()

    #Adding check point to export weight from different epoch
    from keras.callbacks import ModelCheckpoint
    filepath = "noise-weights-improvement-"+str(learnrate)+"-conn-{epoch:02d}.hsf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss',  verbose = 1, save_best_only = False, mode = 'min', period = PERIOD)
    #Import callback function to compute margin
    import my_callbacks
    marginhistory = my_callbacks.Histories()

        #========== Fit model==========
if EXCESSRISK == True:
    print "Saving excessrisk for each epoch"
    history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_test, y_test), callbacks = [checkpoint, marginhistory] )
    acc_history = history_callback.history["acc"]
    val_acc_history = history_callback.history["val_acc"]
    numpy_acc_history = np.array(acc_history)
    numpy_val_acc_history = np.array(val_acc_history)
    accpath = "noise-acc_history_conn_net" + str(learnrate) + ".txt"
    valpath = "noise-val_acc_history_conn_net" + str(learnrate) + ".txt"
    np.savetxt(accpath,numpy_acc_history, delimiter = ",")
else:
    if SAVEMARGIN == True and SAVEWEIGHT == True:
        print "Saving margin and weight for each epoch"
        history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_train, y_train), callbacks = [checkpoint, marginhistory] )
    else:
        if SAVEMARGIN == True:
            print "Saving margin for each epoch"
            history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_train, y_train), callbacks = [marginhistory] )
        else:
            print "Saving weight for each epoch"
            history_callback = model.fit(x_train, y_train, epochs = EPOCHS, batch_size =BATCHSIZE, validation_data = (x_train, y_train), callbacks = [checkpoint] )

# Print out margins after each epoch
if SAVEOUTPUT == True:
    print "Saving output to compute margin distribution"
    print(marginhistory.margins)
    margin_history = marginhistory.margins
    margin_his = np.array(margin_history)
    np.savetxt('noise-margin_history_train_margin.txt', margin_his, delimiter = ",")

    intermediate_output = intermediate_layer_model.predict(x_train)
    final_output = np.array(intermediate_output)
    np.savetxt('noise-final_output.txt', final_output, delimiter = ",")
    np.savetxt('noise-y_train.txt', y_train, delimiter = ",")

