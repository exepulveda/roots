import os.path 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu, threshold_isodata, threshold_li, threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, rectangle
from skimage.color import label2rgb, rgb2gray

from skimage import img_as_float
from skimage.morphology import reconstruction, watershed
from skimage.filters import sobel

from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_otsu, threshold_adaptive

import sys

sys.path += [".."]
from bound import filter_valid_boxes, get_bounding_box,limit_bounding_box,best_box,get_all_boxes

im_path = "/home/esepulveda/Dropbox/Roots videos/Accepted images, video 1.11 - Inverted order/795.tiff"
image = data.imread(im_path)
image = rgb2gray(image)

image = image[10:150,10:150]
image = gaussian_filter(image, 1)

elevation_map = sobel(image)

print np.min(elevation_map),np.max(elevation_map)
#quit()


fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(elevation_map, cmap=plt.cm.jet, interpolation='nearest')
ax.axis('off')
ax.set_title('elevation_map')



plt.show()

markers = np.zeros_like(image)
markers[elevation_map < 0.1] = 0
markers[elevation_map > np.max(elevation_map) * 0.8] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap=plt.cm.jet, interpolation='nearest')
ax.axis('off')
ax.set_title('markers')

plt.show()

segmentation = watershed(elevation_map, markers)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('segmentation')
plt.show()


