import os

import pandas
from matplotlib.pyplot import *
from numpy import *
from numpy.linalg import norm

os.chdir("/h/u16/g2/00/g2lujinn/Downloads/CSC321/Assignment 2")

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    #print x
#     dCdTheta = zeros(theta.shape)
#     for j in range(dCdTheta.shape[0]):
#         for i in range(x.shape[1]):
#             dCdTheta[j] += x[j, i]*(dot(theta.T, x[:,i])-y[i])
#         dCdTheta[j] *= 2
#     
#     return dCdTheta
#     
#     return -2*sum((y-dot(theta.T, x))*x, 1)
    return 2*dot(x, y-dot(theta.T, x))


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    
#     iterations = 0
#     h = array([])
#     v = array([])
    while norm(t - prev_t) >  EPS:
        prev_t = t.copy()
        error = f(x, y, t)/float(x.size)
        t -= alpha*df(x, y, t)
        #break
#         print t, f(x, y, t), df(x, y, t)
#         v = np.append(v, error)
#         iterations +=1
#         h = np.append(h, iterations)
#         if iterations > 2:
#             break
#         #print t, iterations
#     print v
#     print v.size, h.size
#     plt.plot(h, v)
#     plt.xlabel('Iterations')
#     plt.ylabel('Error')
#     plt.title('About as simple as it gets, folks')
#     plt.show()
    return t


dat = pandas.read_csv("galaxy.data")



x1 = dat.loc[:,"east.west"].as_matrix()
x2 = dat.loc[:, "north.south"].as_matrix()
y = dat.loc[:, "velocity"].as_matrix()

x = vstack((x1, x2))
theta = array([-3, 2, 1])

#Check the gradient
h = 0.000001
#print (f(x, y, theta+array([0, h, 0])) - f(x, y, theta-array([0, h, 0])))/(2*h)
#print df(x, y, theta)


theta0 = array([0, 10, 20.])

#larger learning rate leads to trouble!
theta = grad_descent(f, df, x, y, theta0, 0.0000010)
print theta
#Exact solution:  dot(dot(linalg.inv(dot(x, x.T)),x), y)
print dot(dot(linalg.inv(dot(x, x.T)),x), y)
#array([ 1599.7805884 ,     2.32128786,    -3.53935822])
