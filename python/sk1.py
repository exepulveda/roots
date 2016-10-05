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

from bound import filter_valid_boxes, get_bounding_box

for i in [6,8,15,46,38]:
	image = data.imread("/Users/exequiel/projects/roots/python/processing/1.14.AVI/selected/frame-{0}.tiff".format(i))
	image = image[0:100,0:100]
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

	block_size = 35
	gray = rgb2gray(image)
	global_thresh = threshold_otsu(gray)
	print global_thresh
	global_thresh = 0.65
	for t in np.linspace(global_thresh,1.0,20):
		binary_global = gray > t
		#binary_global = gray > 0.90
	
		print "threshold_otsu",threshold_otsu(gray)
		print "threshold_isodata",threshold_isodata(gray)
		print "threshold_li",threshold_li(gray)
		print "threshold_yen",threshold_yen(gray)




		label_img = label(binary_global, connectivity=binary_global.ndim)
		props = regionprops(label_img)
		boxes = []
		for pp in props:
			minr, minc, maxr, maxc = pp.bbox
			boxes += [(minc, minr, maxc - minc, maxr - minr)]


		valid_boxes = filter_valid_boxes(boxes,6,40,14,24,300,900)
		if len(valid_boxes) > 0:
			fig, ax = plt.subplots()
			ax.imshow(binary_global, cmap=plt.cm.gray)
			bb = get_bounding_box(valid_boxes)

			x, y, w, h = bb
			rect = mpatches.Rectangle((x, y), w, h,fill=False, edgecolor='red', linewidth=2)
			ax.add_patch(rect)
			
			print x, y, w, h,w*h
			
			plt.show()