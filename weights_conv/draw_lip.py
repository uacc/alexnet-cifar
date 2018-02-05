import pdb
import os
from conv_weight import LipschitzConstant as lip
import numpy as np

path = os.getcwd()
lip_value = [] 
dtype = [('epochs', int), ('lipschitz', float)]
PaddingMethod = True
for file in os.listdir(path):
    #pdb.set_trace()
    if file[-4:-1] == 'hsf':
        tem = lip(file,PaddingMethod)
        #pdb.set_trace()
        print file
        epoch = file[-8:-5]
        if epoch[0] == '-':
            #pdb.set_trace()
            #epoch = [1:]
            epoch = int(epoch[1:len(epoch)])
        print epoch
        lip_value.append((epoch, tem))
    else:
        continue
lipvalue = np.array(lip_value, dtype = dtype)
lipvalue = np.sort(lipvalue, order = ["epochs"])
pdb.set_trace()
import matplotlib.pyplot as plt
#fig, ax = plt.subplots(nrows = 1, ncols = 1)
#ax.plot(lipvalue['lipschitz'].tolist())
plt.plot(lipvalue["lipschitz"].tolist())
plt.ylabel('lipschitz paddingmethod:'+str(PaddingMethod))
plt.show()
#fig.savefig('lipschitz constant with flatten:'+str(PaddingMethod)+'.jpg')
#plot.close(fig)
