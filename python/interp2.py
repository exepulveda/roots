import numpy as np

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d

import scipy.linalg

from skimage import transform as tf


L = np.loadtxt('../matlab/L.csv',delimiter=","); # image points
Y = np.loadtxt('../matlab/Y.csv',delimiter=","); # image points
w2 = np.loadtxt('../matlab/w.csv',delimiter=","); # image points

lu = scipy.linalg.lu_factor(L)
w = scipy.linalg.lu_solve(lu,Y)

#w = scipy.linalg.solve(L,Y)

np.savetxt('../matlab/wp.csv',w,delimiter=",")

quit()



im = np.loadtxt('../matlab/im.csv',delimiter=","); # image points
ptall = np.loadtxt('../matlab/ptall.csv',delimiter=","); # reference points

imh,imw = im.shape

xd = ptall[:,0].reshape((imh,imw),order="F")
yd = ptall[:,1].reshape((imh,imw),order="F")


im = np.float32(im)

interpoler= RectBivariateSpline(np.arange(imh)+1,np.arange(imw)+1,im)

ret = np.empty(imh*imw)

i = 0
for j in xrange(imh):
    for k in xrange(imw):
        imt = interpoler(yd[j,k],xd[j,k],grid=False)
        #print j+1,k+1,yd[j,k],xd[j,k],imt,im[j,k]
        ret[i] = imt
        i=i+1

im = ret.reshape((imh,imw))

import matplotlib.pyplot as plt

plt.imshow(im,cmap='Greys_r')
plt.show()
