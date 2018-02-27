import os
import numpy as np

train = 'y_train.txt'
output = 'final_output.txt'
noise_train = 'noise-y_train.txt'
noise_output = 'noise-final_output.txt'

with open(train) as file:
    train = [[[int(float(j)) for j in i.split(',')] for i in line.strip().split('\n')] for line in file]

with open(output) as file:
    output = [[[float(j) for j in i.split(',')] for i in line.strip().split('\n')] for line in file]

with open(noise_train) as file:
    noise_train = [[[int(float(j)) for j in i.split(',')] for i in line.strip().split('\n')] for line in file]

with open(noise_output) as file:
    noise_output = [[[int(float(j)) for j in i.split(',')] for i in line.strip().split('\n')] for line in file]

margin = np.zeros(len(train))
noise_margin = np.zeros(len(noise_train))
for i in range(0, len(train)):
    y = np.array(train[i][0])
    predict = np.array(output[i][0])
    
    true_y = np.argmax(y)
    predict_y = np.argmax(predict)

    if true_y == predict_y:
        f_y = predict[predict_y]
        predict[true_y] = 0
        f_i = np.amax(predict)
        margin[i] = f_y - f_i
    else:
        f_y = predict[true_y]
        f_i = predict[predict_y]
        margin[i] = f_y - f_i

for i in range(0, len(noise_train)):
    y = np.array(noise_train[i][0])
    predict = np.array(noise_output[i][0])
    
    true_y = np.argmax(y)
    predict_y = np.argmax(predict)

    if true_y == predict_y:
        f_y = predict[predict_y]
        predict[true_y] = 0
        f_i = np.amax(predict)
        noise_margin[i] = f_y - f_i
    else:
        f_y = predict[true_y]
        f_i = predict[predict_y]
        noise_margin[i] = f_y - f_i
        
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
sns.set(color_codes = True)
np.random.seed(sum(map(ord, "distributions")))
sns.distplot(margin)
sns.distplot(noise_margin)
#plt.show()
plt.savefig('../margin-distri-true-vs-rand.png')
