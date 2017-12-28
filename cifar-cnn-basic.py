from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import read_cifar
from read_cifar import read_input


tf.logging.set_verbosity(tf.logging.INFO)

learning_rate = 0.001
batch_size = 32

input_feature_width = 32
input_feature_height = 32
input_feature_depth = 3

num_labels = 10

conv1_filter_size = 5
conv1_num_filters = 16
conv1_filter_strides = 1

pool1_size = 2
pool1_stride = 2

conv2_filter_size = 5
conv2_num_filters = 20
conv2_filter_strides = 1

pool2_size = 2
pool2_stride = 2

conv3_filter_size = 5
conv3_num_filters = 20
conv3_filter_strides = 1

pool3_size = 2
pool3_stride = 2

logits_size = 4*4*20

graph = tf.Graph()

with graph.as_default():

  

  # define input and output variables and network weights
  tf_train_dataset = tf.placeholder(tf.float32, shape=[None, input_feature_width, input_feature_height, input_feature_depth])
  tf_train_labels = tf.placeholder(tf.float32, shape=[None])
  tf_test_dataset = tf.placeholder(tf.float32, shape=[None, input_feature_width, input_feature_height, input_feature_depth])

  layer1_filter = tf.Variable(tf.truncated_normal([conv1_filter_size, conv1_filter_size, input_feature_depth, conv1_num_filters]))
  layer1_biases = tf.Variable(tf.constant(0.1, shape=[conv1_num_filters]))

  layer2_filter = tf.Variable(tf.truncated_normal([conv2_filter_size, conv2_filter_size, conv1_num_filters, conv2_num_filters]))
  layer2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_num_filters]))

  layer3_filter = tf.Variable(tf.truncated_normal([conv3_filter_size, conv3_filter_size, conv2_num_filters, conv3_num_filters]))
  layer3_biases = tf.Variable(tf.constant(0.1, shape=[conv3_num_filters]))

  logits_weights = tf.Variable(tf.truncated_normal([logits_size, num_labels]))
  logits_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  #define network graph
  h_conv1 = tf.nn.relu(conv2d(tf_train_dataset,layer1_filter) + layer1_biases)
  h_pool1 = max_pool_2x2(h_conv1)

  h_conv2 = tf.nn.relu(conv2d(h_pool1,layer2_filter) + layer2_biases)
  h_pool2 = max_pool_2x2(h_conv2)

  h_conv3 = tf.nn.relu(conv2d(h_pool2,layer3_filter) + layer3_biases)
  h_pool3 = max_pool_2x2(h_conv3)

  h_pool_flat = tf.reshape(h_pool3, [-1, 4*4*20])
  
  # define model output
  logits = tf.matmul(h_pool_flat, logits_weights) + logits_biases

    
  # convert training labels to onehot encoding 
  onehot_labels = tf.one_hot(indices=tf.cast(tf_train_labels, tf.int64), depth=10)

  
  # define training loss
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))
  
  # define optimization algorithm
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  train_prediction = tf.nn.softmax(logits)
  #test_prediction = tf.nn.softmax(cifar_cnn(tf_test_dataset))

  # calculate accuracy
  onehot_train_labels = tf.one_hot(indices=tf.cast(tf_train_labels, tf.int64), depth=10)


  correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1), tf.int64), tf.argmax(onehot_train_labels,1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





[train_all, validate_all, test_all] = read_input()
train_data = np.asarray(train_all['data'])
train_labels = np.asarray(train_all['labels'], dtype=np.int64)


eval_data = np.asarray(test_all['data'])
eval_labels = np.asarray(test_all['labels'], dtype=np.int64)

eval_data_size = eval_labels.shape[0]

# randomly shuffle training data
perm = np.random.permutation(train_data.shape[0])

np.take(train_data, perm, axis=0, out=train_data)
np.take(train_labels, perm, axis=0, out=train_labels)

num_steps = 20000

# start training
with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        minibatch_data = train_data[offset:(offset + batch_size), :, :, :]
        minibatch_labels = train_labels[offset:(offset + batch_size)]
        
        feed_dict = {tf_train_dataset : minibatch_data, tf_train_labels : minibatch_labels}
        
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict = feed_dict
            )
        
        if step % 100 == 0:
            print("Minibatch loss at step {0}: ", l)
            print("Train Accuracy: " , accuracy.eval(feed_dict = {tf_train_dataset: train_data, tf_train_labels: train_labels}))

    print("Test accuracy: ", (accuracy.eval(feed_dict={tf_train_dataset: eval_data, tf_train_labels: eval_labels})))
    




