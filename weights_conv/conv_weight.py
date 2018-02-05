import pdb
import math
import numpy as np
import h5py
def LipschitzConstant(filename, PaddingMethod):
    
    #==========Load input and network structure information==========
    inputsize = [36,36,3]
    num_conv_layers = 3
    padding = 2
    stride = 1
    conv_layers_size = [[inputsize, [32,32,16]],[[16,16,16],[16,16,20]],[[8,8,20],[8,8,20]]]
    # Modify layer size due to padding part
    #conv_layers_size[:,:,0:2] = conv_layers_size[:,:,0:2]+padding*2
    #====================Try to load weights iterativly========== 
    #filename = 'weights-improvement-0.001-conn-19.hsf5'
    f = h5py.File(filename, 'r')
    lip_const = 1

    for i in f.iterkeys():
        print i
        if i == 'model_weights':
            weights = f.get(i)
        else:
            continue
        for k in f.get(i).iterkeys():
            layer = k
            if layer[0:6] == 'conv2d':
                weight_keys = weights.get(layer).get(layer).keys()
                weight_kernel = weights.get(layer).get(layer).get(weight_keys[1])
                weight_bias = weights.get(layer).get(layer).get(weight_keys[0])
                # compute spectral norm from conv layer
                #========== padding conv matrix==========
                kernel = np.array(weight_kernel)
                bias = np.array(weight_bias)
                # padding kernel matrix
                layer_num =int(layer[7])
                layer_size = kernel.shape
                layer_input = np.array(conv_layers_size[layer_num-1][0])
                layer_output = np.array(conv_layers_size[layer_num-1][1])

                input_size = (layer_input[0]+4)*(layer_input[0]+4)*layer_input[2]
                output_size = np.prod(layer_output)

                if PaddingMethod == True:
                    #Asssigning values to A
                    A = np.zeros((output_size, input_size))
                    for entry in range(0, output_size):
                        out_pixel = entry % (layer_output[0]*layer_output[0]) 
                        out_depth = int((entry - out_pixel)/(layer_output[0]*layer_output[0]))
                        out_col = out_pixel % layer_output[0]
                        out_row = int((out_pixel - out_col)/layer_output[0])
                        
                        weight_cur = kernel[:,:,:,out_depth]

                        for depth in range(0,layer_size[2]):
                            for filter_row in range(0, layer_size[0]):
                                   start_input = out_col + (out_row + filter_row)*(layer_input[0]+2) + depth * (layer_input[0]+2) * (layer_input[0]+2)
                                   end_input = start_input + 5
                                   #pdb.set_trace()
                                   try:
                                       A[entry, start_input:end_input] = weight_cur[filter_row,:, depth]
                                   except ValueError:
                                       pdb.set_trace()
                    # padding bias term
                    # join kernel and bias
                else: 
                    # compute Lipschitz constant in other paper
                    d_out = layer_output[2]
                    d_int = layer_input[2]
                    k = (layer_size[0]-1)/2
                    
                    A = np.zeros((d_out, (2*k + 1)*(2*k + 1)*d_int))
                    for depth in range(0, d_out):
                        weight = kernel[:,:,:,depth]
                        #pdb.set_trace()
                        A[depth, :] = math.sqrt((2*k+1)) *np.ravel(weight)



                #==========computing spectral norm from A========== 
                from numpy import linalg as LA
                lip_const = lip_const * LA.norm(A, 2)
            elif layer[0:6]=='dense':
                pdb.set_trace()
                full_key = weights.get(layer).get(layer).keys()
                full_weight = weights.get(layer).get(layer).get(full_key[1])
                full_bias = weights.get(layer).get(layer).get(full_key[0])
                # combine bias term and kernel term
                #

                lip_const = lip_const*LA.norm(full_weight, 2)
            else:
                continue

    return lip_const
