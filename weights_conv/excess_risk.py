import csv
import pdb
import numpy as np
import os
from draw_lip import lip_margin 
def excess_figure_1(NOISE, PaddingMethod):
    #NOISE = True
    # file name
    if NOISE == True:
        acc = 'noise-acc_history_conn_net0.001.txt'
        val = 'noise-val_acc_history_conn_net0.001.txt'
    else:
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

    if NOISE == True:
        np.savetxt("noise_excess_risk.txt", excess_risk, delimiter = ",")
    else:
        np.savetxt("excess_risk.txt", excess_risk, delimiter = ",")
    print excess_risk

    lipepoch, lipschitz, lipvsmargin = lip_margin(NOISE, PaddingMethod)
    epoch = [int(i) for i in lipepoch]
    excess = excess_risk[epoch]
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    #plt.plot(epoch, excess, 'r--', label = 'excess_risk', epoch, lipschitz, 'bs',label = 'lipschitz',  epoch, lipvsmargin, 'g^', label = 'margin')
    plt.plot(epoch, excess, 'r--', label = "excess risk")
    plt.plot(epoch, lipschitz, 'bs', label = "lipschitz")
    plt.plot(epoch, lipvsmargin, 'g^', label = "lipschitz/margin")
    #plt.legend(['Paddingmethod: ' + str(PaddingMethod)])
    if NOISE == True:
        plt.savefig('plot/noise-plot.png')
    else:
        plt.savefig('plot/true-plot.png')

    return excess, lipschitz, lipvsmargin

excesstrue, liptrue, lipvsmargtrue = excess_figure_1(False, True)
excessfalse, lipfalse, lipvsmargfalse = excess_figure_1(False, False)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
epoch = [int(i) for i in range(20)]
plt.plot(epoch, liptrue, 'r--')
plt.plot(epoch, lipfalse, 'g--')
plt.savefig('plot/ture-lipconst-padding-compare.png')

#noise_excesstrue, noise_liptrue, noise_lipvsmargtrue = excess_figure_1(True, True)
noise_excessfalse, noise_lipfalse, noise_lipvsmargfalse = excess_figure_1(True, False)

