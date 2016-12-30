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

        
def visualize_heatmap(W):
    ''' This function is implemented to obtain the heatmaps for the hidden layer
    '''
    fig = figure()
    ax = fig.gca()
    print W.shape
    heatmap = ax.imshow(W[:,799].reshape((64, 64)), cmap= cm.coolwarm)
    fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    show()

# Load network trained from part 1
snapshot = cPickle.load(open("800Hidden.pkl"))
init_W0 = snapshot["W0"]

W0 = tf.Variable(init_W0)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Visualize the hidden features
visualize_heatmap(W0.eval(session=sess))