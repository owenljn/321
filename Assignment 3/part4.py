################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.io import loadmat, savemat
import urllib
from numpy import random

import tensorflow as tf
import cPickle
from actor_classes import class_names

# Initialize the actor names
act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
actor = 'Gerard Butler'

def get_test_for_actor(M, actor):
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 6))
    

    for i in range(6):
        if act[i] == actor:
            break
    test_k =  "test"+str(i)
    batch_xs = vstack((batch_xs, ((array(M[test_k])[:])/255.)  ))
    one_hot = zeros(6)
    one_hot[i] = 1
    #batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k]), 1))   ))
    batch_y_s = vstack((batch_y_s, one_hot))
    return batch_xs, batch_y_s

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 6))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image

x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
i = x_dummy.copy()
i[0,:,:,:] = (imread('training set/'+actor+'/'+'butler12.jpg')[:,:,:3]).astype(float32)
i = i-mean(i)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .layer 1
#         .layer 2
#         .softmax(name='prob'))


net_data = load("../bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''

    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())



x = tf.Variable(i)
# I get a different image by dividing with 255 which looks clearer
x /= 255. 

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)
conv4flatten = tf.reshape(conv4, [1,int(prod(conv4.get_shape()[1:]))])

# Load network from part 2
snapshot = cPickle.load(open("new_snapshot99.pkl"))
init_W0 = snapshot["W0"]
init_b0 = snapshot["b0"]
init_W1 = snapshot["W1"]
init_b1 = snapshot["b1"]

nhid = 300
W0 = tf.Variable(init_W0)
b0 = tf.Variable(init_b0)

W1 = tf.Variable(init_W1)
b1 = tf.Variable(init_b1)

layer1 = tf.nn.tanh(tf.matmul(conv4flatten, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])
gradients = tf.gradients(layer2, x)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
################################################################################

#Output:
# Feed the session with test actor x and its targets
output = sess.run(y)
inds = argsort(output)[0,:]
for i in range(6):
    print class_names[inds[-1-i]], output[0, inds[-1-i]]

# part 5 starts here
gradient = sess.run(gradients)
gradient = np.asarray(gradient)
img = gradient[0][0].flatten()
img[img < 0] = 0
img *= 1/max(img)
plt.imshow(img.reshape((227,227, 3)))
show()



