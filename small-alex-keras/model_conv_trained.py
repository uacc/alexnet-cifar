from __future__ import print_function
import pdb
from glob import glob

from small_alex_keras import BuildModel, importdata()
#import keras
#from keras.callbacks import ModelCheckpoint
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Dense, Dropout, Flatten
#from keras.models import Sequential

from data import get_mnist_data, num_classes, input_shape
from read_activations import get_activations, display_activations

model = BuildModel(FIRST_LAYER_DEPTH, SECOND_LAYER_DEPTH)
    
sgd = optimizers.SGD(lr = LEARNRATE, decay = DECAYFACTOR, momentum = MOMENTUM) 
model.load_weights('../small-alex-keras/data_info/weights-imporvement-conn-04.hsf5')
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
model.summary()
    
#model.compile()
#model.summary()

x_train, y_train, x_test, y_test = importdata()
a = get_activations(model, x_test[0:1], print_shape_only = True, layer_name = 'fc_2')
pdb.set_trace()
