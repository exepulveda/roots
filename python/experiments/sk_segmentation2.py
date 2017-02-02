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

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

import sys

sys.path += [".."]
from bound import filter_valid_boxes, get_bounding_box,limit_bounding_box,best_box,get_all_boxes

im_path = "/home/esepulveda/Dropbox/Roots videos/Accepted images, video 1.11 - Inverted order/319.tiff"
img = data.imread(im_path)
img = rgb2gray(img)

img = img[10:150,10:150]
img = gaussian_filter(img, 1)

graph = image.img_to_graph(img)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
print labels.shape
print labels.reshape(img.shape).shape
label_im = np.empty_like(img)
label_im[:,:] = labels.reshape(img.shape)

plt.matshow(img)
plt.matshow(label_im)

plt.show()
quit()


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


