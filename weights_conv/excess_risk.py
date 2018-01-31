import csv
import pdb
import numpy as np

# file name
acc = 'acc_history_conn_net0.001.txt'
val = 'val_acc_history_conn_net0.001.txt'

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

print excess_risk
