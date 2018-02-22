import pdb
import os
from conv_weight import LipschitzConstant as lip
import numpy as np

def lip_margin():
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
    np.savetxt('noise-lipvalue.txt', lipvalue, delimiter = ",")
#    #Import Lipschitz constant from 1-200 epochs
#     lipsch = 'lipvalue.txt'
#     with open(lipsch) as file:
#         li = [[i for i in line.strip().split('\n')] for line in file]
#     lipvalue = np.zeros(len(li))
#     lipepoch = np.zeros(len(li))
#     for i in range(0, len(li)):
#         tem = li[i][0]
#         tem = tem.split(',')
#         lipvalue[i] = float(tem[1])
#         lipepoch[i] = int(float(tem[0]))
    # Import excess risk to get the range
    excess = 'excess_risk.txt'
    with open(excess) as file:
        exc = [[float(i) for i in line.strip().split('\n')] for line in file]
    excess_risk = np.zeros(len(exc))
    for i in range(0, len(exc)):
        excess_risk[i] = exc[i][0]
    maxrange = np.max(excess_risk)
    minrange = np.min(excess_risk)

    # Resize Lipschitz constant
    #lip_value = lipvalue['lipschitz'] 
    lip_value = lipvalue
    lip_min = np.min(lip_value)
    lip_max = np.max(lip_value)
    lip_value = (maxrange - minrange) * (lip_value - minrange)
    lip_value = lip_value / (lip_max - lip_min)
    lip_value = lip_value + minrange
    
    # Import margin number
    margin = 'margin_history_train_margin.txt'
    with open(margin) as file:
        mar = [[float(i) for i in line.strip().split('\n')] for line in file]
    margin = np.zeros(len(mar))
    for i in range(0, len(mar)):
        margin[i] = mar[i][0]
    # Lip/margin
    #lip_margin = lipvalue['lipschitz']
    lip_margin = lipvalue
    lip_margin = lip_margin[0:len(mar)]

    # Get certain margin number
    margin_value = np.zeros(len(lip_margin))
    for j in range(0, len(lip_margin)):
        #margin_epoch = lipvalue['epochs'][j]
        margin_epoch = int(lipepoch[j])
        margin_value[j] = margin[margin_epoch]
    lip_margin = lip_margin /margin_value 
    # Resize its value
    lm_max = np.max(lip_margin)
    lm_min = np.min(lip_margin)
    lip_margin = (maxrange - minrange) * (lip_margin - minrange)
    lip_margin = lip_margin / (lm_max - lm_min)
    lip_margin = lip_margin + minrange
    return lipepoch, lip_value, lip_margin
