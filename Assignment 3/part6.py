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

# Initialize the actor names and their lastnames for later use
act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'] 
lastname = ['butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon'] 

# Modify the dimension to vectorize the data
train_x = zeros((70, 227,227,3)).astype(float32)
train_y = zeros((1, 6))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

# Set up the directory locations
train_dir = 'training set/'
valid_dir = 'validation set/'
test_dir = 'test set/'
part2_dir = '/home/student/tf_alexnet/part2/'

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
    
def create_new_input(dataset, datasize, n):
    '''This function takes in all images in a given data set with given size and n is used to indicate the index of actor from the act list.
    '''
    actor_dir = act[n]+"/"
    x_dummy = (random.random((datasize,)+ xdim)/255.).astype(float32)
    i = x_dummy.copy()
    if datasize == 70:
        starting_index = 0
    elif datasize == 20:
        starting_index = 70
    else:
        starting_index = 90
    for j in range(datasize): # This loop vectorizes the data
        i[j,:,:,:] = (imread(dataset+actor_dir+lastname[n]+str(starting_index+j)+".jpg")[:,:,:3]).astype(float32)
    i = i-mean(i)
    net_data = load("../bvlc_alexnet.npy").item()

    x = tf.Variable(i)

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
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1],     strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o,    s_h, s_w, padding="SAME", group=group)
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
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1],     strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o,    s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h,  s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    conv4flatten = tf.reshape(conv4, [datasize,            int(prod(conv4.get_shape()[1:]))])
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # The vectorized data needs to be flattened and stored in a numpy array
    new_input = conv4flatten.eval(session=sess)
    return new_input
    
def create_M(train_dir, valid_dir, test_dir):
    ''' This function creates a .mat file which stores all faces
    '''
    mdict = {}
    i = 0
    for actor in act:
        counter = 0
        for filename in os.listdir(test_dir+actor+"/"):
            counter += 1
        train_matrix = create_new_input(train_dir, 70, i)
        valid_matrix = create_new_input(valid_dir, 20, i)
        test_matrix = create_new_input(test_dir, counter, i)
        mdict["train"+str(i)] = train_matrix
        mdict["valid"+str(i)] = valid_matrix
        mdict["test"+str(i)] = test_matrix
        savemat('newfaces.mat', mdict)
        i += 1
        
def create_M_for_actor(actor, test_dir):
    ''' This function creates a .mat file which stores all faces
    '''
    mdict = {}
    counter = 0
    for i in range(6):
        if act[i] == actor:
            break
    for filename in os.listdir(test_dir+actor+"/"):
        counter += 1
    test_matrix = create_new_input(test_dir, counter, i)
    mdict["test"+str(i)] = test_matrix
    savemat(actor+'.mat', mdict)

# Uncomment this line to create the .mat file, this is the most important part.
#create_M(train_dir, valid_dir, test_dir)
M = loadmat("/home/student/tf_alexnet/part2/newfaces.mat")




# The rest part of the code is the same as part 1 except the dimensions of the data
def get_train_batch(M, N):
    n = N/10
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(6):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s

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

def get_valid(M):
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 6))
    
    valid_k =  ["valid"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_train(M):
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        


# Load network from part 2
snapshot = cPickle.load(open("new_snapshot99.pkl"))
init_W0 = snapshot["W0"]
init_b0 = snapshot["b0"]
init_W1 = snapshot["W1"]
init_b1 = snapshot["b1"]

x = tf.placeholder(tf.float32, [None, 64896])
nhid = 300
W0 = tf.Variable(init_W0)
b0 = tf.Variable(init_b0)

W1 = tf.Variable(init_W1)
b1 = tf.Variable(init_b1)

# W0 = tf.Variable(tf.random_normal([64896, nhid], stddev=0.01))
# b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
# 
# W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
# b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])




lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M)
valid_x, valid_y = get_valid(M)

# Initialize the data for plotting
trainCR = array([])
validCR = array([])
testCR = array([])
h = array([])
# Start the training process
for i in range(500):

  batch_xs, batch_ys = get_train_batch(M, 50)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i % 1 == 0:

    valid_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xs, batch_ys = get_train(M)

    # Save parameters for plotting
    trainCR = np.append(trainCR, train_accuracy)
    validCR = np.append(validCR, valid_accuracy)
    testCR = np.append(testCR, test_accuracy)
    h = np.append(h, i)


print "The final performance classification on the training set is: ", train_accuracy


print "The final performance classification on the validation set is: ", valid_accuracy

print "The final performance classification on the test set is: ", test_accuracy


plt.plot(h, trainCR, 'r', label = "training set")
plt.plot(h, validCR, 'g', label = "validation set")
plt.plot(h, testCR, 'b', label = "test set")
plt.title('Correct classification rate vs Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Correct classification rate')
plt.legend(loc='lower right')
plt.show()

################################################################################

# Call create M for actor, change actor's name to see results for actor
actor = 'Gerard Butler'
create_M_for_actor(actor, test_dir)
M_actor = loadmat(actor+'.mat')
for i in range(6):
    if act[i] == actor:
        break
# Call test x function to get test_actorx, test_actory
test_actorx, test_actory = get_test_for_actor(M_actor, actor)

#Output:
# Feed the session with test actor x and its targets
output = sess.run(y, feed_dict={x: test_actorx, y_: test_actory})
inds = argsort(output)[0,:]
for i in range(6):
    print class_names[inds[-1-i]], output[0, inds[-1-i]]



