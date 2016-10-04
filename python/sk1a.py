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
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_otsu, threshold_adaptive

from bound import filter_valid_boxes, get_bounding_box,limit_bounding_box,best_box

for i in range(1000,1400): #7,19,51    18,23,25,34
	im_path = "/Users/exequiel/projects/roots/python/processing/2.42.AVI/accepted/{0}.tiff".format(i)
	if os.path.exists(im_path):
		image = data.imread(im_path)
		image = image[10:80,10:80]
		image = gaussian_filter(image, 1)

		#image = rgb2gray(image)
		#image = img_as_float(image)

		#image = gaussian_filter(image, 1)
		#seed = np.copy(image)
		#seed[1:-1, 1:-1] = image.min()
		#mask = image

		#dilated = reconstruction(seed, mask, method='dilation')
		#im2 = image-dilated
		#print im2
		#binary = im2 > 0
		#plt.imshow(binary, cmap=plt.cm.gray)
		#plt.show()

		#image = data.imread("/Users/exequiel/projects/roots/python/processing/1.15.AVI/selected/frame-54.tiff")
		try:
			fig, ax = plt.subplots()
			bb = best_box(image,debug=True)
	
			ax.imshow(image, cmap=plt.cm.gray)
			if bb is not None:
				x, y, w, h = bb
				rect = mpatches.Rectangle((x, y), w, h,fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)
		
	
			plt.show()
		except:
			pass