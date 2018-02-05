import pdb
import os
from conv_weight import LipschitzConstant as lip
import numpy as np

path = os.getcwd()
lip_value = [] 
dtype = [('epochs', int), ('lipschitz', float)]
PaddingMethod = True
for file in os.listdir(path):
    if file[-4:-1] == 'hsf':
        tem = lip(file,PaddingMethod)
        print file
        epoch = file[-8:-5]
        if epoch[0] == '-':
            epoch = int(epoch[1:len(epoch)])
        print epoch
        lip_value.append((epoch, tem))
    else:
        continue
# Sort lip constant using epochs
lipvalue = np.array(lip_value, dtype = dtype)
lipvalue = np.sort(lipvalue, order = ["epochs"])

# Import excess risk to get the range
import os
excess = 'excess_risk.txt'
with open(excess) as file:
    exc = [[float(i) for i in line.strip().split('\n')] for line in file]
excess_risk = np.zeros(len(exc))
for i in range(0, len(exc)):
    excess_risk[i] = exc[i][0]
maxrange = np.max(excess_risk)
minrange = np.min(excess_risk)

# Resize Lipschitz constant
#
#


# Import margin number
margin = 'mh.txt'
with open(margin) as file:
    mar = [[float(i) for i in line.strip().split('\n')] for line in file]
margin = np.zeros(len(mar))
for i in range(0, len(mar)):
    margin[i] = mar[i][0]

# Get margin and excess risk for certain epoch and plot together
import matplotlib.pyplot as plt
plt.plot(lipvalue["lipschitz"].tolist())
plt.ylabel('lipschitz paddingmethod:'+str(PaddingMethod))
plt.show()
