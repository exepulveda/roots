import os.path
import numpy as np

from skimage import data
from skimage.feature import match_template
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

def load_templates(path):
	ret = [None]*10
	for i in range(10):
		tmp_fn = os.path.join(path,"{0}.tiff".format(i))
		if os.path.exists(tmp_fn):
			image = data.imread(tmp_fn)
			gray = rgb2gray(image)
			ret[i] = gray

	return ret

def filter_valid_boxes(boxes,min_width,max_width,min_height,max_height,min_a,max_a,debug=False):
	#first filter height
	if debug: print "boxes",boxes
	if debug: print "min-max",min_width,max_width,min_height,max_height,max_a
	
	filter1 = []
	for x,y,w,h in boxes:
		if debug: print "filter1:boxes",w,h,w*h
		if min_height <= h <= max_height and min_width <= w <= max_width and min_a <= w*h <= max_a:
			filter1 += [(x,y,w,h)]
	
	if debug: print "filter1",filter1
	#print "filter2",filter2
	
	return filter1
	
def distance(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)
	
def get_bounding_box(boxes,debug=False,max_distance=25.0):
	#boxes are already filtered
	#order according area
	n = len(boxes)
	if n > 1:
		area = []
		for i,(x,y,w,h) in  enumerate(boxes):
			area += [(w*h,i)]
		area.sort()
		b1 = boxes[area[-1][1]]
		b2 = boxes[area[-2][1]]
		
		x1,y1,w1,h1 = b1
		x2,y2,w2,h2 = b2
		
		d = distance(x1,y1,x2,y2)
		if debug: print d
		if d > max_distance:
			return b1
		else:
			x = min(x1,x2)
			y = min(y1,y2)
			xf = max(x1+w1,x2+w2)
			yf = max(y1+h1,y2+h2)
		
			return x,y,xf-x,yf-y
	else:
		return boxes[0]

def limit_bounding_box(boxes,limit=30,debug=False):
	above_limit = []
	under_limit = []
	for bb in boxes:
		x,y,w,h = bb
		if w >= limit:
			above_limit += [(x,y,w,h)]
		else:
			under_limit += [(x,y,w,h)]
	if len(above_limit) > 0:
		return above_limit
	else:
		return under_limit
		
def get_target_bounding_box(box,tw,th):
	#middle point
	x,y,w,h = box
	xm = x + w/2.0
	ym = y + h/2.0
	
	return int(xm - tw/2.0),int(ym - th/2.0),tw,th
	
def best_box(image,global_thresh=None,steps=20,limit=30,debug=False):
	ret = []
	gray = rgb2gray(image)
	if global_thresh is None:
		global_thresh = threshold_otsu(gray)
	for t in np.linspace(global_thresh,1.0,steps):
		if debug: print "best_box:t:",t
		binary_global = gray > t

		label_img = label(binary_global, connectivity=binary_global.ndim)
		props = regionprops(label_img)
		boxes = []
		for pp in props:
			minr, minc, maxr, maxc = pp.bbox
			boxes += [(minc, minr, maxc - minc, maxr - minr)]

		valid_boxes = filter_valid_boxes(boxes,8,40,15,28,200,900,debug=debug)
		if len(valid_boxes) > 0:
			bb = get_bounding_box(valid_boxes,debug=debug)
			
			x, y, w, h = bb
			
			if debug: print "best_box:valid_boxes:",x,y,w,h
			ret += [(w>limit,w*h,bb)]
			
	ret.sort()
	print ret
	if len(ret) > 0:
		return ret[-1][2]
	else:
		return None
	
def match(image,template):
	print image.shape,template.shape
	result = match_template(image, template)
	
	#find i,j with maxvalue
	maxrow = np.argmax(result,axis=1)
	print maxrow
	maxcol = np.argmax(result,axis=0)
	print maxcol
	#maxcorrel = result[maxrow,maxcol]
	return np.max(result)