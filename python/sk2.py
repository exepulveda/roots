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

from bound import filter_valid_boxes, get_bounding_box, get_target_bounding_box, load_templates, match

templates = load_templates("/Users/exequiel/projects/roots/python/models/templates")

i = 48
image = data.imread("/Users/exequiel/projects/roots/python/processing/1.14.AVI/selected/frame-{0}.tiff".format(i))
image = image[:100,:100]
image = gaussian_filter(image, 1)

gray = rgb2gray(image)
global_thresh = 0.65
for t in np.linspace(global_thresh,1.0,20):
	binary_global = gray > t

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
		bb = get_target_bounding_box(bb,40,30)
		
		for k,template in enumerate(templates):
			corr = match(binary_global,template)
			print k,corr
		x, y, w, h = bb
		rect = mpatches.Rectangle((x, y), w, h,fill=False, edgecolor='red', linewidth=2)
		ax.add_patch(rect)
			
		
		plt.show()