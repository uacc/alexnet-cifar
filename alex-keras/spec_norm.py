import numpy as np
from numpy import linalg as LA
import pdb
from spicy import *
# #Load weights trained before
# weight1 = np.load('weight_conv_1_filters.npy')
# weight2 = np.load('weight_conv_2_filters.npy')
# weight3 = np.load('weight_conv_3_filters.npy')
# weight4 = np.load('weight_conv_4_filters.npy')
# weight5 = np.load('weight_conv_5_filters.npy')
# 
# weight6 = np.load('weight_full_1.npy')
# weight7 = np.load('weight_full_2.npy')
# weight8 = np.load('weight_full_3.npy')

# Calculate the marginal bound using l2 norm of X and prediction distribution from previous code


#General information about each layer 
l1_norm = np.zeros((8,1))
spec_norm = np.zeros((8,1))
layer_io = np.array([[227,227,3], [55, 55, 96],[27,27,96],[27,27,256],[13,13,256],[13,13,384],
    [13,13,384],[13,13,384],[13,13,384],[13,13,384]])
pad_info = np.array([0, 2, 1, 1, 1])
stride_info = np.array([4, 1, 1, 1, 1])


#Calculate norm 
for i in range(0,8):
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
        input_image_size = (layer_io[2*i][0]  + 2 * pad) * (layer_io[2*i][0]  + 2 * pad)
        input_row = (layer_io[2*i][0]  + 2 * pad)
        input_col = input_row
        input_dim = layer_io[2*i][2]
        output_dim = layer_io[2*i+1][0] * layer_io[2*i+1][1] * layer_io[2*1+1][2]
        output_image_size = layer_io[2*i+1][0]*layer_io[2*i+1][1]
        peak = layer_io[2*i+1][0]^2
        
        # # New padding weight matrix (using numpy array)
        # pad_weight = np.zeros((output_dim,input_dim))
        # print(output_dim, input_dim)
        # # each pixel in the output
        # for j in range(0, output_dim):
        #     # denote which pixel we are calculating in the output image
        #     pixel = j%(layer_io[2*i+1][0] * layer_io[2*i+1][1])
        #     # depth of current pixel
        #     depth = int((j - j%pixel) / j)
        #     # get the filter for current depth
        #     filter_weight = weight[:,:,:,depth]
        #     (x, y, z) = filter_weight.shape
        #     
        #     # for the input image we are using which entries
        #     length_row = (layer_io[2*i][0] - x + 2*pad) / stride + 1 #how many filters in each row
        #     if length_row != layer_io[2*i+1][0]:
        #         pdb.set_trace()
        #     start_col = (pixel % length_row) * stride
        #     start_row = int(pixel / length_row)
        #     
        #     # Assign value to each entries
        #     for k in range(0,x):
        #         for n in range(0,y):
        #             for m in range(0, z):
        #                 # calculate the current input ind in the vector
        #                 ind = start_ind + k + input_row * n + input_row * input_col * m
        #                 if ind >= input_dim:
        #                     pdb.set_trace()
        #                 pad_weight[j, ind] = filter_weight[k,n,m]
        #                 
        #     # Make sure we are going well for each steps
        #     if j % peak == 0:
        #         print(depth)
        # spec_norm[i] = LA.norm(pad_weight, 2)
        # l1_norm[i] = LA.norm(pad_weight, 'nuc')
        
        
        # New padding weight matrix (using sparse matrix)
        # We need to define its index with row and column and data
        filter_size = x * y * z * depth
        row = np.zeros((filter_size * output_dim, 1))
        col = np.zeros((filter_size * output_dim, 1))
        data = np.zeros((filter_size * output_dim, 1))
        
    	# Assign value to row
        for i in range(0, output_dim):
            row[(i*filter_size):(i*(filter_size+1))-1] = i
            cur_ind = i*filter_size
            out_dep = cell(i/output_image_size)
            out_index = i%output_image_size
            out_col = out_index%layer_io[2*i+1][0]
            out_row = cell(out_index/layer_io[2*i+1][09
            
            # Calculate the index we will need in the input space
            inp_col = out_col*x-stride[i]
            inp_row = out_row*y-stride[i]
            weight2d = weight[:,:,:,out_dep]
            # We will need to repeat for input_depth many times
            for j in range(input_depth):
                col[cur_ind:cur_ind+x]
                col[cur_ind:cur_ind+x+]

    else:
        # Fully connected layers
        weight_file = 'weight_full_' + str(i) + '.npy'
        weight = np.load(weight_file)
        spec_norm[i] = LA.norm(weight, 2)
        l1_norm[i] = LA.norm(weight, 'nuc')
        

    
    






















