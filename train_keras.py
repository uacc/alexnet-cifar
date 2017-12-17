from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
#from KerasLayers.Custom_layers import LRN2D

# import data and resize it
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

# Global Constants
NB_CLASS=1000
LEARNING_RATE=0.01
MOMENTUM=0.9
ALPHA=0.0001
BETA=0.75
GAMMA=0.1
DROPOUT=0.5
WEIGHT_DECAY=0.0005
#LRN2D_NORM=True
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'


def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=WEIGHT_DECAY):
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    #if lrn2d_norm:
    #   x=LRN2D(alpha=ALPHA,beta=BETA)(x)

    return x


def create_model():
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,227,227)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(227,227,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering: '+str(DIM_ORDERING))

    # Convolution Net Layer 1
    x=conv2D_lrn2d(img_input,96,(11,11),4,padding='valid')
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='valid',data_format=DATA_FORMAT)(x)

    # Convolution Net Layer 2
    x=conv2D_lrn2d(x,256,(5,5),1,padding='same')
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='valid',data_format=DATA_FORMAT)(x)

    # Convolution Net Layer 3~5
    x=conv2D_lrn2d(x,384,(3,3),1,padding='same')
    x=conv2D_lrn2d(x,384,(3,3),1,padding='same')
    x=conv2D_lrn2d(x,256,(3,3),1,padding='same')
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='valid',data_format=DATA_FORMAT)(x)

    # Convolution Net Layer 6
    x=Flatten()(x)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(DROPOUT)(x)

    # Convolution Net Layer 7
    x=Dense(4096,activation='relu')(x)
    x=Dropout(DROPOUT)(x)

    # Convolution Net Layer 8
    x=Dense(output_dim=NB_CLASS,activation='softmax')(x)

    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


def check_print():
    # Create the Model
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT=create_model()

    # Create a Keras Model
    model=Model(input=img_input,output=[x])
    model.summary()

    # Save a PNG of the Model Build
    #plot_model(model,to_file='AlexNet.png')

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    print 'Model Compiled'
    return model

if __name__=='__main__':
    model = check_print()
    model.fit(x_train, y_train, epochs = 6)