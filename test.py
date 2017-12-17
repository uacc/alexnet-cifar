import numpy as np
from numpy import linalg as LA
import pdb

l1_norm = np.zeros((8,1))
spec_norm = np.zeros((8,1))
layer_io = np.array([[227,227,3], [55, 55, 96],[27,27,96],[27,27,256],[13,13,256],[13,13,384],
    [13,13,384],[13,13,384],[13,13,384],[13,13,384]])
pad_info = np.array([0, 2, 1, 1, 1])
stride_info = np.array([4, 1, 1, 1, 1])


#Calculate norm 
for i in range(0,1):
#     Loading current layer weights
    if i <= 5:
        weight_file = 'weight_conv_'+ str(i) + '_filters.npy'
        weight = np.load(weight_file)
        (x,y,z,depth) = weight.shape
        
        #Padding the weight matrix
        pad = pad_info[i]
        stride = stride_info[i]
        
        # input and output vector size
        # input vector need padding info
        input_dim = (layer_io[2*i][0]  + 2 * pad)* (layer_io[2*i][1] + 2 * pad) * layer_io[2*i][2]
        input_row = (layer_io[2*i][0]  + 2 * pad)
        input_col = input_row
        output_dim = layer_io[2*i+1][0] * layer_io[2*i+1][1] * layer_io[2*1+1][2]
        peak = layer_io[2*i+1][0]^2
        
        # New padding weight matrix
        print(output_dim, input_dim)
        pad_weight = np.zeros((output_dim,input_dim))         # each pixel in the output
        for j in range(0, output_dim):
            # denote which pixel we are calculating in the output image
            pixel = j%(layer_io[2*i+1][0] * layer_io[2*i+1][1])
            # depth of current pixel
            depth = int((j - j%pixel) / j)
            # get the filter for current depth
            filter_weight = weight[:,:,:,depth]
            (x, y, z) = filter_weight.shape
            
            # for the input image we are using which entries
            length_row = (layer_io[2*i][0] - x + 2*pad) / stride + 1 #how many filters in each row
            if length_row != layer_io[2*i+1][0]:
                pdb.set_trace()
            start_col = (pixel % length_row) * stride
            start_row = int(pixel / length_row)
            
            # Assign value to each entries
            for k in range(0,x):
                for n in range(0,y):
                    for m in range(0, z):
                        # calculate the current input ind in the vector
                        ind = start_ind + k + input_row * n + input_row * input_col * m
                        if ind >= input_dim:
                            pdb.set_trace()
                        pad_weight[j, ind] = filter_weight[k,n,m]
                        
            # Make sure we are going well for each steps
            if j % peak == 0:
                print(depth)
        spec_norm[i] = LA.norm(pad_weight, 2)
        l1_norm[i] = LA.norm(pad_weight, 'nuc')
    else:
        # Fully connected layers
        weight_file = 'weight_full_' + str(i) + '.npy'
        weight = np.load(weight_file)
        spec_norm[i] = LA.norm(weight, 2)
        l1_norm[i] = LA.norm(weight, 'nuc')
        



















