import pdb
import tensorflow as tf
import numpy as np
import pickle
import sys

n_classes=10
image_width=32
image_height=32
image_depth=3

n_validate_samples=2000
n_test_samples=5

train_all={'data':[], 'labels':[]}
validate_all={'data':[], 'labels':[]}
test_all={'data':{}, 'labels':[]}
label_names_for_validation_and_test=None


def unpickle(file):
    with open(file, 'rb') as fo:
        #pdb.set_trace()
        #dict=pickle.load(fo, encoding='bytes')
        dict=pickle.load(fo)
    return dict

def prepare_input(data=None, labels=None):
    global image_height, image_width, image_depth
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    #do mean normaization across all samples
    mu = np.mean(data, axis=0)
    mu = mu.reshape(1,-1)
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    #data is transformed from (no_of_samples, 192) to (no_of_samples , image_height, image_width, image_depth)
    #make sure the type of the data is no.float32
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    return data, labels

def read_input():
    trn_all_data=[]
    trn_all_labels=[]
    vldte_all_data=[]
    vldte_all_labels=[]
    tst_all_data=[]
    tst_all_labels=[]

    #test_temp=unpickle('C:/Users/pranjal/Downloads/cifar-10/cifar-10-batches-py/test_batch')
    test_temp = unpickle('./cifar-10-batches-py/test_batch')
    #print(sys.getsizeof(trn_all_data))


    d=unpickle('./cifar-10-batches-py/data_batch_1')
    trn_all_data.append(d[b'data'])
    trn_all_labels.append(d[b'labels'])
    #print(sys.getsizeof(trn_all_data))

    d=unpickle('./cifar-10-batches-py/data_batch_2')
    trn_all_data.append(d[b'data'])
    trn_all_labels.append(d[b'labels'])

    d=unpickle('./cifar-10-batches-py/data_batch_3')
    trn_all_data.append(d[b'data'])
    trn_all_labels.append(d[b'labels'])
   
    d=unpickle('./cifar-10-batches-py/data_batch_4')
    trn_all_data.append(d[b'data'])                  
    trn_all_labels.append(d[b'labels'])

    d=unpickle('./cifar-10-batches-py/data_batch_5')     
    trn_all_data.append(d[b'data'])                  
    trn_all_labels.append(d[b'labels'])

    trn_all_data, trn_all_labels = (np.concatenate(trn_all_data).astype(np.float32), np.concatenate(trn_all_labels).astype(np.int64))

    #print(trn_all_data.shape)


    vldte_all_data=test_temp[b'data'][0:(n_validate_samples+n_test_samples), :]
    vldte_all_labels=test_temp[b'labels'][0:(n_validate_samples+n_test_samples)]
    vldte_all_data, vldte_all_labels =  (np.concatenate([vldte_all_data]).astype(np.float32),np.concatenate([vldte_all_labels]).astype(np.int64))

    train_all['data'], train_all['labels'] = prepare_input(data=trn_all_data, labels=trn_all_labels)

    validate_and_test_data, validate_and_test_labels = prepare_input(data=vldte_all_data, labels=vldte_all_labels)

    validate_all['data'] = validate_and_test_data[0:n_validate_samples, :, :, :]
    validate_all['labels'] = validate_and_test_labels[0:n_validate_samples]
    test_all['data'] = validate_and_test_data[n_validate_samples:(n_validate_samples+n_test_samples), :, :, :]
    test_all['labels'] = validate_and_test_labels[n_validate_samples:(n_validate_samples+n_test_samples)]

    return train_all, validate_all, test_all

    #print(test_all['data'].shape)




