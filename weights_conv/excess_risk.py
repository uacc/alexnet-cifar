import csv
import pdb
import numpy as np
import os
from draw_lip import lip_margin 

# file name
#acc = 'acc_history_conn_net0.001.txt'
#val = 'val_acc_history_conn_net0.001.txt'
acc = 'noise-acc_history_conn_net0.001.txt'
val = 'noise-val_acc_history_conn_net0.001.txt'

# read acc and val_acc value
with open(acc) as file:
    acc = [[float(i) for i in line.strip().split('\n')] for line in file]
    #pdb.set_trace()
with open(val) as file:
    val = [[float(i) for i in line.strip().split('\n')] for line in file]

# Compute the excess risk
excess_risk = np.zeros(len(acc))
for i in range(0, len(acc)):
    excess_risk[i] = acc[i][0] - val[i][0]

#np.savetxt("excess_risk.txt", excess_risk, delimiter = ",")
print excess_risk

lipepoch, lipschitz, lipvsmargin = lip_margin()
pdb.set_trace()
epoch = [int(i) for i in lipepoch]
excess = excess_risk[epoch]
import matplotlib.pyplot as plt
#plt.plot(epoch, excess, 'r--', label = 'excess_risk', epoch, lipschitz, 'bs',label = 'lipschitz',  epoch, lipvsmargin, 'g^', label = 'margin')
plt.plot(epoch, excess, 'r--', label = "excess risk")
plt.plot(epoch, lipschitz, 'bs', label = "lipschitz")
plt.plot(epoch, lipvsmargin, 'g^', label = "lipschitz/margin")
plt.legend()
plt.show()

