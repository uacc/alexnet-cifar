#!/usr/bin/env python

#Import Data
from keras.datasets import cifar10
(x, y_train), (x_test, y_test) = cifar10.load_data()
print('Data Imported')

#Resize Image
import PIL
from PIL import Image
import numpy as np
i = 0
x_train = np.zeros((x.shape[0],227,227,3))
for img in x:
    img = Image.fromarray(img.astype('uint8'),'RGB')
    im = img.resize((227,227),resample = PIL.Image.BILINEAR)
    tem = np.asarray(im).reshape((227,227,3))
    x_train[i] = tem
    if i%1000==0:
        print(i)
    i = i + 1
print('End of resize')

#Save datafor future use
np.save('x_train', x_train)
np.save('y_train', y_train)
print('Save resized data')