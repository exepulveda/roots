import numpy as np
import cv2
from rbfwarp2d import rbfwarp2d,wrap, homo

from morphing import Morpher

import matplotlib.pyplot as plt

image_filename = '../matlab/im.jpg'

#load correspondences 
ps = np.loadtxt('../matlab/ps.txt',delimiter=","); # image points
pd = np.loadtxt('../matlab/pd.txt',delimiter=","); # reference points

original = cv2.imread(image_filename)
im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

#mor = Morpher(ps-1, pd-1,regularization=100.0,subsampling_factor=1.0)
#rect = mor.interpolate(im)
#plt.imshow(rect,cmap='Greys_r')
#plt.show()

#quit()


#rect = homo(im, ps-1.0, pd-1.0)

#quit()



#plt.imshow(im,cmap='Greys_r')
#plt.show()


h,w = im.shape

rectified = rbfwarp2d(im, ps, pd, method = "g",r=.5*w)
#rectified = wrap(im, ps, pd)

print rectified.shape

plt.imshow(rectified,cmap='Greys_r')
plt.show()


'''
% interpolate
im = rgb2gray(original);
w = size(im,2);
rectified = rbfwarp2d( im, ps, pd,'gau',10*w);

% plot
subplot(1,2,1)
imshow(original)
title('Orininal image');
hold on
plot( ps(:,1),ps(:,2),'r*' );

subplot(1,2,2)
imshow(uint8(rectified));
title('Thin-plate warping');
hold on
plot( ps(:,1),ps(:,2),'r*' );
plot( pd(:,1),pd(:,2),'gO' );
'''
