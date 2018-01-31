import pdb
import numpy as np
import h5py
#==========Load input and network structure information==========
inputsize = [36,36,3]
num_conv_layers = 3
padding = 2
stride = 1
conv_layers_size = [[inputsize, [32,32,16]],[[16,16,16],[16,16,20]],[[8,8,20],[8,8,20]]]
# Modify layer size due to padding part
#conv_layers_size[:,:,0:2] = conv_layers_size[:,:,0:2]+padding*2
#====================Try to load weights iterativly========== 
filename = 'weights-improvement-0.001-conn-19.hsf5'
f = h5py.File(filename, 'r')

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
            
            
            #Construct spectral norm term for current A
            #First check whether we got correct A matrix
            pdb.set_trace()
            
            #unrolled l_1 vector norm
            tem = np.absolute(A)
            print 'l1 norm of A'
            print np.sum(tem)
            
            #spectral norm
            print 'spectral norm of A'
            from numpy import linalg as LA
            print LA.norm(A, 2)

        else:
            continue
            


#----------Lipschitz---------- 
#----------Lipschitz/margin---------- 
