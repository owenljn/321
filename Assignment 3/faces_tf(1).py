from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat, savemat


t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)


act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'] 

import tensorflow as tf


# This function flattens the image, and returns a 60x4096 array
def get_digit_matrix(img_dir):
    img_files = sorted([img_dir + filename for filename in os.listdir(img_dir)])
    img_shape = array(imread(img_files[0])).shape[:2] # open one image to get the size 
    img_matrix = array([imread(img_file)[:,:,0].flatten() for img_file in img_files])
    img_matrix = array([img_matrix[i,:]/(norm(img_matrix[i,:])+0.0001) for i in range(img_matrix.shape[0])])
    return img_matrix

# Note each training directory stores 70 faces, each validation and test directory stores 20 faces.
train_dir = '/home/student/tf_alexnet/training set/'
valid_dir = '/home/student/tf_alexnet/validation set/'
test_dir = '/home/student/tf_alexnet/test set/'

def create_M(train_dir, valid_dir, test_dir):
    ''' This function creates a .mat file which stores all faces
    '''
    mdict = {}
    i = 0
    for actor in act:
        train_matrix = get_digit_matrix(train_dir+actor+"/")
        valid_matrix = get_digit_matrix(valid_dir+actor+"/")
        test_matrix = get_digit_matrix(test_dir+actor+"/")
        mdict["train"+str(i)] = train_matrix
        mdict["valid"+str(i)] = valid_matrix
        mdict["test"+str(i)] = test_matrix
        savemat('faces.mat', mdict)
        i += 1

create_M(train_dir, valid_dir, test_dir)
M = loadmat("/home/student/tf_alexnet/faces.mat")

def get_train_batch(M, N):
    n = N/10
    batch_xs = zeros((0, 64*64))
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
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_valid(M):
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    valid_k =  ["valid"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_train(M):
    batch_xs = zeros((0, 64*64))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        



x = tf.placeholder(tf.float32, [None, 4096])


nhid = 300
W0 = tf.Variable(tf.random_normal([4096, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

# snapshot = cPickle.load(open("/home/student/tf_alexnet/snapshot50.pkl"))
# W0 = tf.Variable(snapshot["W0"])
# b0 = tf.Variable(snapshot["b0"])
# W1 = tf.Variable(snapshot["W1"])
# b1 = tf.Variable(snapshot["b1"])

# W0 = np.random.rand(4096, 6)
# b0 = np.random.rand(6, 1)
# W1 = np.random.rand(nhid, 6)
# b1 = np.random.rand(nhid, 1)

layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

W = tf.Variable(tf.random_normal([4096, 6], stddev=0.01))
b = tf.Variable(tf.random_normal([6], stddev=0.01))
layer = tf.matmul(x, W)+b

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

trainCR = array([])
validCR = array([])
testCR = array([])
h = array([])
for i in range(5000):
  #print i  
  batch_xs, batch_ys = get_train_batch(M, 50)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 1 == 0:
    #print "i=",i
    valid_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

    #print "valid:", valid_accuracy
    
    #print "Test:", test_accuracy
    batch_xs, batch_ys = get_train(M)

    #print "Train:", train_accuracy
    #print "Penalty:", sess.run(decay_penalty)
    trainCR = np.append(trainCR, train_accuracy)
    validCR = np.append(validCR, valid_accuracy)
    testCR = np.append(testCR, test_accuracy)
    h = np.append(h, i)
    #print valid_accuracy, test_accuracy, train_accuracy

    # snapshot = {}
    # snapshot["W0"] = sess.run(W0)
    # snapshot["W1"] = sess.run(W1)
    # snapshot["b0"] = sess.run(b0)
    # snapshot["b1"] = sess.run(b1)
    #cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))

print "The final performance classification on the test set is: ", test_accuracy
plt.plot(h, trainCR, 'r', label = "training set")
plt.plot(h, validCR, 'g', label = "validation set")
plt.plot(h, testCR, 'b', label = "test set")
plt.title('Correct classification rate vs Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Correct classification rate')
plt.legend(loc='lower right')
plt.show()


#print "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
#batch_xs, batch_ys = get_train(M)
#print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
#print "Penalty:", sess.run(decay_penalty)