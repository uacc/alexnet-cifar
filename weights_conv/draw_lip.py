import pdb
import os
from conv_weight import LipschitzConstant as lip
import numpy as np

path = os.getcwd()
lip_value = np.zeros(1)

for file in os.listdir(path):
    pdb.set_trace()
    if file[-4:-1] == 'hsf':
        tem = lip(file, True)
        lip_value = lip_value.append(tem)
    else:
        continue

pdb.set_trace()
