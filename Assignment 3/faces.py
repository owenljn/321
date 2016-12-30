import numpy as np
from l2_distance import l2_distance
import os
from pylab import *

def knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    count = zeros((1, len(valid_labels)))
    for i in range (1,7):
        count_i = np.sum(valid_labels==i, axis=1).reshape(1,len(valid_labels))
        count = vstack((count, count_i))
    valid_labels = np.argmax(count, axis=0)
    return valid_labels.T

# This function flattens the image, and returns a 60x1024 array
def get_digit_matrix(img_dir):
    img_files = sorted([img_dir + filename for filename in os.listdir(img_dir) if filename[-4:] == ".jpg"])
    img_shape = array(imread(img_files[0])).shape[:2] # open one image to get the size 
    img_matrix = array([imread(img_file)[:,:,0].flatten() for img_file in img_files])
    img_matrix = array([img_matrix[i,:]/(norm(img_matrix[i,:])+0.0001) for i in range(img_matrix.shape[0])])
    return img_matrix

def hitrate(super_valid):
    answer = zeros((len(super_valid),1))
    for i in range(0,6):
        answer[range(10*i, 10*i+10)] = i+1
    result = super_valid-answer
    hits = len(result[result==0])
    return double(hits)/len(super_valid)

def male_hitrate(super_valid):
    answer = zeros((len(super_valid),1))
    for i in range(0,6):
        if ((i == 1) or (i==4) or (i==5)):
            answer[range(10*i, 10*i+10)] = 1
        else:
            answer[range(10*i, 10*i+10)] = 0
    male = np.logical_or(super_valid == 2, super_valid == 5, super_valid == 6)
    result = answer - male
    hits = len(result[result==0])
    return double(hits)/len(super_valid)
    
def female_hitrate(super_valid):
    answer = zeros((len(super_valid),1))
    for i in range(0,6):
        if ((i == 0) or (i==2) or (i==3)):
            answer[range(10*i, 10*i+10)] = 1
        else:
            answer[range(10*i, 10*i+10)] = 0
    female = np.logical_or(super_valid == 1, super_valid == 3, super_valid == 4)
    result = answer - female
    hits = len(result[result==0])
    return double(hits)/len(super_valid)
        
# First set up the image matrix for training, validation and test sets
training_dir = '/h/u16/g2/00/g2lujinn/Downloads/CSC321/part2/training set/'
valid_dir = '/h/u16/g2/00/g2lujinn/Downloads/CSC321/part2/validation set/'
test_dir = '/h/u16/g2/00/g2lujinn/Downloads/CSC321/part2/test set/'

# Note that training set matrix has a dimension of 600x1024 and validation set
# has a dimension of 60x1024
img_matrix = get_digit_matrix(training_dir)
valid_matrix = get_digit_matrix(valid_dir)
test_matrix = get_digit_matrix(test_dir)
# Now use a loop to create 6 training labels for each actor/actress, and call knn
# for each of the actor/actress
# a list of Ks are to be tested for knn algorithm
K = [1, 2, 5, 10, 20, 50, 80, 100, 150, 200]
for k in K:
    print 'k is now: ',k
    # Initialize a training label matrix for all actors/actresses
    training_label = zeros((600,1))
    for i in range(0,6):
        training_label[range(100*i, 100*i+100)] = i+1
    valid_label = knn(k, img_matrix, training_label, img_matrix)
    valid_label = valid_label.reshape(len(valid_label), 1)
    print hitrate(valid_label)

"""
After the validation test, it can be seen that when k = 1, the hitrate is the
highest, therefore, I choose k=1 for test set and training set itself
"""
k = 1
# Pick k = 1 and apply it on test set
training_label = zeros((600,1))
for i in range(0,6):
    training_label[range(100*i, 100*i+100)] = i+1
valid_label = run_knn(k, img_matrix, training_label, test_matrix)
valid_label = valid_label.reshape(len(valid_label), 1)
print 'The hitrate of the test set for k = 1 is:', hitrate(valid_label)
    
# For part 5
K = [1, 2, 5, 10, 20, 50, 80, 100, 150, 200]
for k in K:
    training_label = zeros((600,1))
    for i in range(0,6):
        training_label[range(100*i, 100*i+100)] = i+1
    valid_label = run_knn(k, img_matrix, training_label, valid_matrix)
    valid_label = valid_label.reshape(len(valid_label), 1)
    print 'K =', k
    #print 'The hitrate of the test set for k = 1 is:', hitrate(valid_label)
    print 'The male hitrate is:', male_hitrate(valid_label)
    print 'The female hitrate is:', female_hitrate(valid_label)
    