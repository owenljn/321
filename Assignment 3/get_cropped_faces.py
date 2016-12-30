from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from hashlib import sha256
#from rgb2gray import rgb2gray


# Instructions to run the code:
# the two paths I used below are my local paths, "uncropped/" and "cropped/" folders
# should be created at the same location where this python file exists. The code
# will download the images automatically and it's implemented the way such that
# the gray scale images are generated and cropped right after the image is
# downloaded. "faces.txt" file is a txt file which contains all info from
# "subset_actors.txt" and "subset_actresses.txt", so the code can handle all
# required actors/actresses at one time.



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            

path1 = '/home/student/tf_alexnet/part2/uncropped/'
path2 = '/home/student/tf_alexnet/part2/cropped/'
# First loop through the follwing actors' images in uncropped folder
act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'] 
gray()
for a in act:
    name = a.split()[1].lower()
    i = 0
    # This faces.txt contains all actors and actresses
    for line in open("/home/student/tf_alexnet/part2/faces.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            x1 = int(line.split()[5].split(',')[0])
            y1 = int(line.split()[5].split(',')[1])  
            x2 = int(line.split()[5].split(',')[2]) 
            y2 = int(line.split()[5].split(',')[3])
            correctHash = line.split()[6]
            timeout(testfile.retrieve, (line.split()[4], "part2/uncropped/"+filename), {}, 30)
            if not os.path.isfile("part2/uncropped/"+filename):
                continue
            else:
                # Filter out the corrupted images
                file = open("part2/uncropped/"+filename, "rb").read()
                fileHash = sha256(file).hexdigest()
                if fileHash != correctHash:
                    continue
                try:
                    # Now crop the image at each loop
                    j = imread("part2/uncropped/"+filename)
                    # Crop the image at each loop and call rgb2gray function
                    out = j[y1:y2, x1:x2]
                    # Resize the image and save it
                    out = imresize(out, [227, 227])
                    imsave("part2/cropped/"+filename, out)
                except: # Handle the unexpected runtime errors
                    continue
            
            i += 1