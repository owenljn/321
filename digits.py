from pylab import *
from numpy import *
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
from scipy.io import loadmat

#Load the MNIST digit data, this path should be changed
M = loadmat("J:/Users/OwenLJN/Desktop/CSC321Projects/Project2/mnist_all.mat")
#Display the 150-th "5" digit from the training set
#imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
#show()

# part 1 code
# This function plots 10 images for each of the 10 digits in one figure
def plot_number():
    fig = plt.figure()
    for i in range(0, 10):
        digit_name = "train" + str(i)
        for j in range(0, 10):
            plt.subplot(10,10,10*i+j+1)
            number = M[digit_name][j*10].reshape((28,28))
            imgplot = plt.imshow(number, cmap=cm.gray)
    plt.show()
#plot_number()

# part 2 code
def compute_softmax(x, W, b):
    '''Return the softmax matrix of any given 28x28 input image, note this 
    neural network has no hidden layers. So W and b are initialized as zeros'''

    # calculate outputs
    o = compute_y(x, W, b)
    # calculate the softmax matrix
    return softmax(o)

# part 3 code
def gradient_Wb(x, W, b, y, y_):
    '''Return the gradient of the cost function with respect to W and b. Note
    that the neural network used in part 3 is the same one from part 2'''

    # gradient of cost function is p-y_, where y_ is a 10xM one-hot encoding
    # vector, and p is the softmax of y

    # Use the result of the example showed in "opt.pdf", page 7/7

    p = softmax(y)
    dCdy = p - y_

    dCdW = dot(x, dCdy.T)
    dCdb = dot(dCdy, ones((dCdy.shape[1], 1)))
    return dCdW, dCdb

# part 4 code
def finite_difference_approx(x, W, b, y_):
    '''Return the gradient w.r.t W and b using the finite difference
    approximation method'''
    # setup the differences
    h = 0.001
    dW = zeros((784, 10))
    dW[:] = h
    db = zeros((10, 1))
    db[:] = h
    
    #Estimate the derivative w.r.t W, b
    # Using the example showed in "opt.pdf", page 7/7
    dCdW = (cost(softmax(compute_y(x, W+dW, b)), y_) - \
    cost(softmax(compute_y(x, W-dW, b)), y_))/(2*dW)
    
    dCdb = (cost(softmax(compute_y(x, W, b+db)), y_) - \
    cost(softmax(compute_y(x, W, b-db)), y_))/(2*db)
    return dCdW, dCdb

def compute_y(x, W, b):
    '''Return the output y, where y is an 10x1 matrix'''
    return dot(W.T, x)+b


# part 4
def grad_descent(M, W, b, m, learning_rate, flag):
    '''Returns the correct gradients of W and b, it also chooses which gradient 
    method to run.
    
    Note:   this is the vectorized version of batch linear regression, 
            it takes around 1m40s to run.
    Inputs:
        M:           The MNIST dataset which contains the images to be processed.
        W:           The 784x10 weight matrix.
        b:           The 10x1 bias matrix.
        m:           The batch size.
        learning_rate:The rate initialized for computing gradients
        flag:        The variable used to indicate which gradient method to run.
    Outputs:
        W, b: The correctly computed Weight and bias matrices.
    '''
    
    W_grad = zeros((784, 10))
    b_grad = zeros((10, 1))
    # Iterate through some validation set for all the 10 digits
    for iteration in range(0, 200):
        for i in range(0, 10):
            x = vstack(M["train"+str(i)][0:m]/255.0).T # x is 784xm matrix
            y = compute_y(x, W, b) # y is now a 10xm matrix
            y_ = zeros((10, m))
            y_[i, :] = 1
            if flag == 1:
                W_temp, b_temp = gradient_Wb(x, W, b, y, y_)
            else:
                W_temp, b_temp = finite_difference_approx(x, W, b, y_)
            W_grad += W_temp
            b_grad += b_temp
            # Update the gradients at each iteration
            W -= learning_rate*(W_grad)/m
            b -= learning_rate*(b_grad)/m
    return W, b

 # part 5
def mini_batch_grad_descent(M, W, b, batch_size, learning_rate):
    '''Returns the correct gradients of W and b.
    
    Note:   this is the vectorized version of batch linear regression, 
            it takes around 1m40s to run.
    Inputs:
        M:             The MNIST dataset which contains the images to be processed.
        W:             The 784x10 weight matrix.
        b:             The 10x1 bias matrix.
        batch_size:    The mini-batch size
        learning_rate: The rate initialized for computing gradients
    Outputs:
        W, b: The correctly computed Weight and bias matrices.
        The learning curve graph for hitrate/Error vs Iterations
    '''

    W_grad = zeros((784, 10))
    b_grad = zeros((10, 1))
#     h = array([])
#     v = array([])
    # Iterate through some validation set for all the 10 digits
    for iteration in range(0, 200):
        #error = 0
        for i in range(0, 10):
            size = M["train"+str(i)].size/784
            # Use a random integer to choose where to start for mini-batch
            index = np.random.randint(0, size-batch_size)
            x = vstack(M["train"+str(i)][index:index+batch_size]/255.0).T # x is 784xm matrix
            y = compute_y(x, W, b) # y is a 10xm matrix
            y_ = zeros((10, batch_size))
            y_[i, :] = 1
            W_temp, b_temp = gradient_Wb(x, W, b, y, y_)
            W_grad += W_temp
            b_grad += b_temp
            W -= learning_rate*(W_grad)/batch_size
            b -= learning_rate*(b_grad)/batch_size
            #error += sqrt(sum((compute_softmax(x, W, b) - y_)**2))
        #Record the rate data/error data
#         rate = hitrate(W, b)
#         v = np.append(v, rate)
#         v = np.append(v, error)
#         h = np.append(h, iteration)
#         plt.plot(h, v)
    #plot the Error vs Iterations graph
#     plt.xlabel('Iterations')
#     plt.ylabel('Error')
#     plt.title('Error learning curve vs Iterations')
#     plt.show()
#     plt.xlabel('Iterations')
#     plt.ylabel('Hitrate')
#     plt.title('Hitrate learning curve vs Iterations')
#     plt.show()
    return W, b
    
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    #print tile(sum(exp(y),0), (len(y),1))
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    #if you don't want tanh at the top layer
    #L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def cost(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, ((1- L1**2)*dCdL1).T )
    
def compare_gradient():
    '''Print out the difference in gradient computation with function of part 3 
    and a finite difference approximation function for the same set of data'''
    # Initiate the data for comparison
    x = M["test5"][150].T/255.0
    x = x.reshape((784, 1))
    W = np.random.rand(784, 10)
    W /= W.size
    b = zeros((10, 1))
    learning_rate = 0.001


    my_dCdW, my_dCdb = grad_descent(M, W, b, 200, learning_rate, 1)
    f_dCdW, f_dCdb = grad_descent(M, W, b, 200, learning_rate, 0)
#     print "The hitrate is: "
#     print hitrate(my_dCdW, my_dCdb), "and", hitrate(f_dCdW, f_dCdb) 
    print "The number to be predicted is: ", 5
    print "The number predicted by the normal method is:"
    print np.argmax(softmax(compute_y(x, my_dCdW, my_dCdb)))
    print "The number predicted by the finite-diff approx method is:"
    print np.argmax(softmax(compute_y(x, f_dCdW, f_dCdb)))
    print "--------------------------------------------------------------------"
    print "The precise gradient w.r.t W is:\n"
    print my_dCdW
    print "--------------------------------------------------------------------"
    print "The precise gradient w.r.t b is:\n"
    print my_dCdb
    print "--------------------------------------------------------------------"
    print "The approximate gradient w.r.t b is:\n"
    print f_dCdW
    print "--------------------------------------------------------------------"
    print "The approximate gradient w.r.t b is:\n"
    print f_dCdb
    print "--------------------------------------------------------------------"
    print "The difference of dCdW is:\n", my_dCdW-f_dCdW
    print "--------------------------------------------------------------------"
    print "The sum difference of dCdW is:\n", sum(my_dCdW-f_dCdW)
    print "--------------------------------------------------------------------"
    print "The difference of dCdb is:\n", my_dCdb-f_dCdb
#compare_gradient() 

def hitrate(dCdW, dCdb):
    '''Return the classification rate for any given W, b gradient
    This function calculates the hitrate by using all available test cases.'''

    hit = 0.0
    total = 0.0

    # Iterate through some test set for all the 10 digits
    for i in range(0, 10):
        size = M["test"+str(i)].size/784
        # Vectorize the matrix to get rid of the loop
        digit_matrix = argmax(compute_softmax(x, dCdW, dCdb), axis = 0)
        digit_matrix = digit_matrix.reshape((1, size))
        
        targets = zeros((digit_matrix.shape[0], digit_matrix.shape[1]))
        targets[:] = i
        
        result = digit_matrix - targets
        hit += result.shape[1] - np.count_nonzero(result)
        total += digit_matrix.shape[1]

    return hit/total

def plot_rateVSiteration():
    '''This function plots the hitrate vs iteration learning curve, it can 
    also be modified to produce the images for part 5'''
    # Initiate the data for comparison
    W = np.random.rand(784, 10)
    W /= W.size
    b = zeros((10, 1))
    learning_rate = 0.001
    m = 50
    dCdW, dCdb = mini_batch_grad_descent(M, W, b, m, learning_rate)
    # use two lists to grab 20 correctly classified and 10 incorrectly classifed
    # digits
    max_count_correct = []
    # max_count_incorrect = []
    # incorrect_digits = []
    # Iterate through some test set for all the 10 digits
    # The loop is needed to gather individual information for digits
    for i in range(0, 10):
        counter = 0
        size = M["test"+str(i)].size/784
        for j in range(0, size):
            x = (M["test"+str(i)][j]/255.0).T
            x = x.reshape((784, 1))
            digit = argmax(compute_softmax(x, dCdW, dCdb))
            # Grab the disired images
            if digit - i == 0 and len(max_count_correct) < 20 and counter < 2:
                max_count_correct.append((i, j))
                counter += 1
            # if digit - i != 0 and len(max_count_incorrect) < 10:
            #     max_count_incorrect.append((i, j))
            #     incorrect_digits.append(digit)

    for item in max_count_correct:
    # for item in max_count_incorrect:
        imshow(M["train"+str(item[0])][item[1]].reshape((28,28)), cmap=cm.gray)
        # print "The incorrectly predicted digit is: ", incorrect_digits[0]
        show()
#plot_rateVSiteration()

# Part 6
def visualize_heatmap():
    '''This function is implemented to obtain the heatmaps for each digit.'''
    # call gradient descent to get the dCdW

    # Initiate the data for comparison
    batch_size  = 50 # set m to 5000 to obtain accurate gradients
    W = np.random.rand(784, 10)
    W /= W.size
    b = zeros((10, 1))
    learning_rate = 0.01

    dCdW, dCdb = mini_batch_grad_descent(M, W, b, batch_size, learning_rate)
    for i in range(0, 10):
        fig = figure(1)
        ax = fig.gca()    
        heatmap = ax.imshow(dCdW[:,i].reshape((28,28)), cmap = cm.coolwarm)    
        fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        show()

#visualize_heatmap()

# Part 7
def compute_p_hidden(x, W0, b0, W1, b1):
    '''Return only the final output of the network'''
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    #L1 = dot(W1.T, L0) + b1
    return softmax(L1)

def gradient_Hidden(x, W0, b0, W1, b1, L0, y, p, y_):
    '''Return the gradient of the cost function with respect to W and b. Note
    that the neural network used in part 3 is the same one from part 2'''

    # gradient of cost function is p-y_, where y_ is a 10xM one-hot encoding
    # vector, and p is the softmax of y

    # Use the result of the example showed in "opt.pdf", page 7/7
    dCdy = p - y_
    dCdL0 = dot(W1, (1 - y**2)*dCdy)

    dCdW1 = dot(L0, ((1 - y**2)*dCdy).T) # compute gradient w.r.t hidden layer 1
    dCdW0 = dot(x, ((1-L0**2)*dCdL0).T) # compute gradient w.r.t hidden layer 0
    dCdb0 = dot(dCdL0, ones((dCdL0.shape[1], 1)))
    dCdb1 = dot(dCdy, ones((dCdy.shape[1], 1)))
    return dCdW0, dCdW1, dCdb0, dCdb1

def finite_difference_approx_hidden(x, W0, b0, W1, b1, y_):
    '''Return the gradient w.r.t W and b using the finite difference
    approximation method'''
    # setup the differences
    h = 0.001
    dy = zeros((10, 300))
    dy[:] = h
    dL0 = zeros((300, 300))
    dL0[:] = h
    dW0 = zeros((784, 300))
    dW0[:] = h
    dW1 = zeros((300, 10))
    dW1[:] = h
    db0 = zeros((300, 1))
    db0[:] = h
    db1 = zeros((10, 1))
    db1[:] = h

    #Estimate the derivative w.r.t W, b
    # Using the example showed in "opt.pdf", page 7/7
    dCdW0 = (cost(softmax(compute_p_hidden(x, W0+dW0, b0, W1, b1)), y_) - \
            cost(softmax(compute_p_hidden(x, W0-dW0, b0, W1, b1)), y_) )/(2*dW0)

    dCdW1 = (cost(softmax(compute_p_hidden(x, W0, b0, W1-dW1, b1)), y_) - \
            cost(softmax(compute_p_hidden(x, W0, b0, W1-dW1, b1)), y_) )/(2*dW1) 
    
    dCdb0 = (cost(softmax(compute_p_hidden(x, W0, b0, W1, b1)), y_) - \
            cost(softmax(compute_p_hidden(x, W0, b0, W1, b1)), y_))/(2*db0)

    dCdb1 = (cost(softmax(compute_p_hidden(x, W0, b0, W1, b1+db1)), y_) - \
            cost(softmax(compute_p_hidden(x, W0, b0, W1, b1-db1)), y_))/(2*db1)

    return dCdW0, dCdW1, dCdb0, dCdb1

def grad_descent_hidden(M, m, learning_rate, flag):
    '''Returns the correct gradients of W0, W1 and b0, b1, it also chooses which gradient 
    method to run, this function uses a neural network with a hidden layer.
    
    Note:   this is the vectorized version of batch linear regression, 
            it takes around 1m40s to run.
    Inputs:
        M:           The MNIST dataset which contains the images to be processed.
        m:           The batch size. Now it's initialized as 300
        learning_rate:The rate initialized for computing gradients
        flag:        The variable used to indicate which gradient method to run.
    Outputs:
        W, b: The correctly computed Weight and bias matrices.
    '''
    W0 = init_W0
    W1 = init_W1
    b0 = init_b0
    b1 = init_b1
    W0_grad = zeros((784, W0.shape[1]))
    b0_grad = zeros((b0.shape[0], 1))    
    W1_grad = zeros((W1.shape[0], W1.shape[1]))
    b1_grad = zeros((b1.shape[0], 1))
    # Iterate through some validation set for all the 10 digits
    for iteration in range(0, 200):
        for i in range(0, 10):
            x = vstack(M["train"+str(i)][0:m]/255.).T # x is 784xm matrix
            y_ = zeros((10, m))
            y_[i, :] = 1
            if flag == 1: # Choose which method to run
                L0, y, p = forward(x, W0, b0, W1, b1) # compute the hidden layer and softmax p
                W0_temp, W1_temp, b0_temp, b1_temp = \
                gradient_Hidden(x, W0, b0, W1, b1, L0, y, p, y_)
            else:
                W0_temp, W1_temp, b0_temp, b1_temp = \
                finite_difference_approx_hidden(x, W0, b0, W1, b1, y_)
            W0_grad += W0_temp
            b0_grad += b0_temp
            W1_grad += W1_temp
            b1_grad += b1_temp
            # Update the gradients at each iteration
            W0 -= learning_rate*(W0_grad)/m
            b0 -= learning_rate*(b0_grad)/m
            W1 -= learning_rate*(W1_grad)/m
            b1 -= learning_rate*(b1_grad)/m
    return W0, b0, W1, b1


def hitrate_hidden(dCdW0, dCdb0, dCdW1, dCdb1):
    '''Return the classification rate for any given W, b gradient
    This function calculates the hitrate by using all available test cases.'''

    hit = 0.0
    total = 0.0

    # Iterate through some test set for all the 10 digits
    for i in range(0, 10):
        size = M["test"+str(i)].size/784
        x = (M["test"+str(i)][0:size]/255.0).T
        # Vectorize the hitrate computation
        digit_matrix = argmax(compute_p_hidden(x, dCdW0, dCdb0, dCdW1, dCdb1), axis = 0)
        digit_matrix = digit_matrix.reshape((1, size))
        
        targets = zeros((digit_matrix.shape[0], digit_matrix.shape[1]))
        targets[:] = i
        
        result = digit_matrix - targets
        hit += result.shape[1] - np.count_nonzero(result)
        total += digit_matrix.shape[1]
    return hit/total
    
# Part 8
# Initialize data
snapshot = cPickle.load(open("J:/Users/OwenLJN/Desktop/CSC321Projects/Project2/snapshot50.pkl"))
init_W0 = snapshot["W0"]
init_b0 = snapshot["b0"].reshape((300,1))
init_W1 = snapshot["W1"]
init_b1 = snapshot["b1"].reshape((10,1))
def compare_gradient_hidden():
    '''Print out the difference in gradient computation with function of part 3 
    and a finite difference approximation function for the same set of data'''
    # Initiate the data for comparison
    learning_rate = 0.001

    my_dCdW0, my_dCdb0, my_dCdW1, my_dCdb1 = grad_descent_hidden(M, 200, learning_rate, 1)
    print "----------------------------------------------------------------------"
    print "The hitrate is: "
    print hitrate_hidden(my_dCdW0, my_dCdb0, my_dCdW1, my_dCdb1)
    print "--------------------------------------------------------------------"
    f_dCdW0, f_dCdb0, f_dCdW1, f_dCdb1 = grad_descent_hidden(M, 200, learning_rate, 0)
    print "--------------------------------------------------------------------"
    print "The hitrate is: "
    print hitrate_hidden(f_dCdW0, f_dCdb0, f_dCdW1, f_dCdb1)
    print "--------------------------------------------------------------------"
    print "The precise gradient w.r.t W is:\n"
    print my_dCdW0, my_dCdW1
    print "--------------------------------------------------------------------"
    print "The precise gradient w.r.t b is:\n"
    print my_dCdb0, my_dCdb1
    print "--------------------------------------------------------------------"
    print "The approximate gradient w.r.t b is:\n"
    print f_dCdW0, f_dCdW1
    print "--------------------------------------------------------------------"
    print "The approximate gradient w.r.t b is:\n"
    print f_dCdb0, f_dCdb1
    print "--------------------------------------------------------------------"
    print "The difference of dCdW0 is:\n", my_dCdW0-f_dCdW0
    print "--------------------------------------------------------------------"
    print "The difference of dCdW1 is:\n", my_dCdW1-f_dCdW1
    print "--------------------------------------------------------------------"
    print "The sum difference of dCdb0 is:\n", sum(my_dCdb0-f_dCdb0)
    print "--------------------------------------------------------------------"
    print "The sum difference of dCdb1 is:\n", sum(my_dCdb1-f_dCdb1)
#compare_gradient_hidden()

#Part 9
def mini_batch_grad_descent_hidden(M, W0, b0, W1, b1, batch_size, learning_rate):
    '''Returns the correct gradients of W and b.
    
    Note:   this is the vectorized version of batch linear regression, 
            it takes around 1m40s to run.
    Inputs:
        M:             The MNIST dataset which contains the images to be processed.
        batch_size:    The mini-batch size
        learning_rate: The rate initialized for computing gradients
    Outputs:
        W, b: The correctly computed Weight and bias matrices.
    '''

    W0_grad = zeros((784, W0.shape[1]))
    b0_grad = zeros((b0.shape[0], 1))    
    W1_grad = zeros((W1.shape[0], W1.shape[1]))
    b1_grad = zeros((b1.shape[0], 1))
    h = array([])
    v = array([])
    # Iterate through some validation set for all the 10 digits
    for iteration in range(0, 20000): # Iterations of 20000 achieves 95% hitrate
        error = 0
        for i in range(0, 10):
            size = M["train"+str(i)].size/784
            # Use a random integer to choose where to start for mini-batch
            index = np.random.randint(0, size-batch_size)
            x = vstack(M["train"+str(i)][index:index+batch_size]/255.).T # x is 784xm matrix
            y_ = zeros((10, batch_size))
            y_[i, :] = 1
            L0, y, p = forward(x, W0, b0, W1, b1) # compute the hidden layer and softmax p
            W0_temp, W1_temp, b0_temp, b1_temp = \
                gradient_Hidden(x, W0, b0, W1, b1, L0, y, p, y_)
            W0_grad += W0_temp/batch_size
            b0_grad += b0_temp/batch_size
            W1_grad += W1_temp/batch_size
            b1_grad += b1_temp/batch_size
            # Update the gradients at each iteration
            W0 -= learning_rate*(W0_grad)
            b0 -= learning_rate*(b0_grad)
            W1 -= learning_rate*(W1_grad)
            b1 -= learning_rate*(b1_grad)
            error += sqrt(sum((compute_p_hidden(x, W0, b0, W1, b1) - y_)**2))
        #Record the rate data/error data
        #rate = hitrate_hidden(W0, b0, W1, b1)
        #v = np.append(v, rate)
        v = np.append(v, error)
        h = np.append(h, iteration)
        plt.plot(h, v, 'r-')
    #plot the Error vs Iterations graph
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error learning curve vs Iterations')
    plt.show()
    # plt.xlabel('Iterations')
    # plt.ylabel('Hitrate')
    # plt.title('Hitrate vs Iterations learning curve')
    # plt.show()
    return W0, b0, W1, b1

def test_part9():
    '''This function plots the hitrate vs iteration learning curve, it can 
    also be modified to produce the images for part 5'''
    # Initiate the data for comparison
    m = 50
    W0 = init_W0
    W1 = init_W1
    b0 = init_b0
    b1 = init_b1
    learning_rate = 0.00001 # currently the best is about 0.0004

    W0, b0, W1, b1 = mini_batch_grad_descent_hidden(M, W0, b0, W1, b1, m, learning_rate)

    # calculate hitrate
    print hitrate_hidden(W0, b0, W1, b1)

    # use two lists to grab 20 correctly classified and 10 incorrectly classifed
    # digits
    #max_count_correct = []
    max_count_incorrect = []
    incorrect_digits = []
    # Iterate through some test set for all the 10 digits
    # The loop is needed to gather individual information for digits
    for i in range(0, 10):
        size = M["test"+str(i)].size/784
        counter = 0
        for j in range(0, size):
            x = (M["test"+str(i)][j]/255.0).T
            x = x.reshape((784, 1))
            digit = argmax(compute_p_hidden(x, W0, b0, W1, b1))
            # Grab the disired images
            # if digit - i == 0 and len(max_count_correct) < 20 and counter < 4:
            #     max_count_correct.append((i, j))
            #     counter += 1
            if digit - i != 0 and len(max_count_incorrect) < 20 and counter < 2:
                max_count_incorrect.append((i, j))
                incorrect_digits.append(digit)
                counter += 1

    # for item in max_count_correct:
    i = 0
    for item in max_count_incorrect:
        imshow(M["train"+str(item[0])][item[1]].reshape((28,28)), cmap=cm.gray)
        print "The incorrectly predicted digit is: ", incorrect_digits[i]
        i += 1
        show()

#test_part9()

# Part 10

def visualize_heatmap_hidden():
    '''This function is implemented to obtain the heatmaps for each digit.'''
    # call gradient descent to get the dCdW

    # Initiate the data for comparison
    m = 50
    W0 = init_W0
    W1 = init_W1
    b0 = init_b0
    b1 = init_b1
    learning_rate = 0.001

    W0, b0, W1, b1  = mini_batch_grad_descent_hidden(M, W0, b0, W1, b1, m, learning_rate)
    print hitrate_hidden(W0, b0, W1, b1)
    for n in range(0, W0.shape[1]):
        print W1[n,:] # Print out the value of the weights of W1
        print argmax(W1[n,:]) # Print out the maximum possible digit's weight
        fig = figure(1)
        ax = fig.gca()    
        heatmap = ax.imshow(W0[:,n].reshape((28,28)), cmap = cm.coolwarm)    
        fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        show()

#visualize_heatmap_hidden()

#Load sample weights for the multilayer neural network
# snapshot = cPickle.load(open("/h/u16/g2/00/g2lujinn/Downloads/CSC321/Assignment 2/snapshot50.pkl"))
# W0 = snapshot["W0"]
# b0 = snapshot["b0"].reshape((300,1))
# W1 = snapshot["W1"]
# b1 = snapshot["b1"].reshape((10,1))
# print "W0's shape is: ", b0.shape
# print "W1's shape is: ", b1.shape
#Load one example from the training set, and run it through the
#neural network
# x = M["train5"][148:149].T    
#L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
#y = argmax(output)
#print output
#print y
################################################################################
#Code for displaying a feature from the weight matrix mW
# fig = figure(1)
# ax = fig.gca()    
# heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
# fig.colorbar(heatmap, shrink = 0.5, aspect=5)
# show()
################################################################################

