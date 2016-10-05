import os.path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import stats

from skimage import data
from skimage.feature import match_template
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
from skimage.filters import threshold_isodata


MIN_WIDTH = 8
MAX_WITH  = 40
MIN_HEIGHT = 17
MAX_HEIGHT = 28
MIN_AREA   = 200
MAX_AREA   = 900
TWO_DIGIT_LIMIT   = 31

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
	
def get_bounding_box(boxes,min_a,max_a,debug=False,max_distance=25.0):
    #boxes are already filtered
    #order according area
    n = len(boxes)
    if n > 1:
        area = []
        for i,(x,y,w,h) in  enumerate(boxes):
            if debug: print "get_bounding_box",i,x,y,w,h,w*h
            if min_a <= w*h <= max_a:
                area += [(w*h,i)]
                
        area.sort()
        if len(area) > 1:
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
            return boxes[area[-1][1]]
    else:
        return boxes[0]

def limit_bounding_box(boxes,limit=TWO_DIGIT_LIMIT,debug=False):
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
	
	return max(0,int(xm - tw/2.0)),max(0,int(ym - th/2.0)),tw,th
	
def best_box(image,global_thresh=None,steps=20,limit=TWO_DIGIT_LIMIT,debug=False):
    ret = []
    gray = rgb2gray(image)
    if global_thresh is None:
        global_thresh = threshold_otsu(gray)

        #global_thresh = threshold_isodata (gray)       
        #global_thresh = threshold_li(gray)
        #global_thresh = threshold_otsu(gray)
        #global_thresh = threshold_yen(gray)
        
        
    for t in np.linspace(global_thresh,1.0,steps):
        if debug: print "best_box:t:",t
        binary_global = gray > t

        label_img = label(binary_global, connectivity=binary_global.ndim)
        props = regionprops(label_img)
        boxes = []
        for pp in props:
            minr, minc, maxr, maxc = pp.bbox
            boxes += [(minc, minr, maxc - minc, maxr - minr)]

        valid_boxes = filter_valid_boxes(boxes,MIN_WIDTH,MAX_WITH,MIN_HEIGHT,MAX_HEIGHT,MIN_AREA,MAX_AREA,debug=debug)
        if len(valid_boxes) > 0:
            bb = get_bounding_box(valid_boxes,MIN_AREA,MAX_AREA,debug=debug)
            
            x, y, w, h = bb
            
            if debug: print "best_box:valid_boxes:",x,y,w,h
            ret += [(w>limit,w*h,bb)]
            
    ret.sort()
    if debug: print ret
    if len(ret) > 0:
        return ret[-1][2]
    else:
        return None
	
def get_all_boxes(image,global_thresh=None,steps=20,limit=TWO_DIGIT_LIMIT,debug=False):
    ret = []
    gray = rgb2gray(image)
    if global_thresh is None:
        #global_thresh = threshold_otsu(gray)

        #global_thresh = threshold_isodata (gray)       
        #global_thresh = threshold_li(gray)
        global_thresh = threshold_otsu(gray)
        #global_thresh = threshold_yen(gray)
        
        
    for t in np.linspace(global_thresh,1.0,steps):
        if debug: print "best_box:t:",t
        binary_global = gray > t

        label_img = label(binary_global, connectivity=binary_global.ndim)
        props = regionprops(label_img)
        boxes = []
        for pp in props:
            minr, minc, maxr, maxc = pp.bbox
            boxes += [(minc, minr, maxc - minc, maxr - minr)]

        valid_boxes = filter_valid_boxes(boxes,MIN_WIDTH,MAX_WITH,MIN_HEIGHT,MAX_HEIGHT,MIN_AREA,MAX_AREA,debug=debug)
        if len(valid_boxes) > 0:
            bb = get_bounding_box(valid_boxes,MIN_AREA,MAX_AREA,debug=debug)
            
            x, y, w, h = bb
            
            if debug: print "best_box:valid_boxes:",x,y,w,h
            ret += [(w>limit,w*h,bb)]
            
    ret.sort()
    return ret
    
    
def match(image,template,debug=False):
	if debug: print image.shape,template.shape
	result = match_template(image, template)
	
	#find i,j with maxvalue
	maxrow = np.argmax(result,axis=1)
	if debug: print maxrow
	maxcol = np.argmax(result,axis=0)
	if debug: print maxcol
	#maxcorrel = result[maxrow,maxcol]
	return np.max(result)
    
def match_digit(image,templates,min_w=24,is_two_digits=True,debug=False):
    h,w = image.shape
    
    if is_two_digits:
        #first digit should be: 1,2,3,4,5
        ret = []
        for digit in [1,2,3,4,5]:
            result = match_template(image[:,:w/2], templates[digit])
            max_correl = np.max(result)
            ret += [(max_correl,digit)]
            if debug: print "digit1:",digit,max_correl
        #second digit should be any except if first digit is 5. That case options are 0,1,2,3,4
        if len(ret) > 0:
            ret.sort()
            #best digit is the last
            correl1,digit1 = ret[-1]
        else:
            return None,None

        digits = range(0,10) if digit1 != 5 else [0,1,2,3,4]
        ret = []
        for digit in digits:
            result = match_template(image[:,w - w/2:], templates[digit])
            max_correl = np.max(result)
            ret += [(max_correl,digit)]
            if debug: print "digit2:",digit,max_correl
        if len(ret) > 0:
            ret.sort()
            #best digit is the last
            correl2,digit2 = ret[-1]
        else:
            return None,None
            
        return (correl1,digit1),(correl2,digit2)

    else:
        #assert h >= 20, "problem with w {0}:{1}".format(h,20)
        #assert w >= 18, "problem with h {0}:{1}".format(w,18)
        
        ret = []
        #one digit: 6,7,8,9
        for digit in [6,7,8,9]:
            result = match_template(image, templates[digit])
            max_correl = np.max(result)
            if debug: print "digit1:",digit,max_correl
            ret += [(max_correl,digit)]
        
        if len(ret) > 0:
            ret.sort()
            #best digit is the last
            return ret[-1],None
            
def predict_mode(image_name,templates,tw=44,th=28,min_w=24,debug=False):
    image = data.imread(image_name)
    image = image[10:80,10:80]
    image = gaussian_filter(image, 1)
    
    
    boxes = get_all_boxes(image,debug=debug)
    
    #predict all boxes
    predictions = []
    for is_two_digits,area,bb in boxes:
        if bb is None:
            return None
            
        x, y, w, h = bb
        
        if is_two_digits:
            tw = 44
        else:
            tw = 36 
        
        tb = get_target_bounding_box(bb,tw,th)
        
        assert tb[2] <= tw, "problem with tw {0}:{1}".format(tb[2],tw)
        assert tb[3] <= th, "problem with th {0}:{1}".format(tb[3],th)
        
        if debug: print "target box",tb,"is_two_digits",is_two_digits

        if tb is not None:
            x, y, w, h = tb
            x1 = x
            x2 = x + w - max(tw/2,min_w)
            w1 = max(tw/2,min_w)
            w2 = max(tw/2,min_w)

            selection = image[y:y+h,x:x+w]
            selection = rgb2gray(selection)
            prediction = match_digit(selection,templates,min_w=min_w,is_two_digits=is_two_digits,debug=debug)
            
            if is_two_digits and prediction[0] is not None and prediction[1] is not None:
                prediction = prediction[0][1] * 10 + prediction[1][1]
            elif not is_two_digits and prediction[0] is not None:
                prediction = prediction[0][1]
            else:
                prediction = None

            if prediction:
                if debug: print prediction,bb
                predictions += [prediction]

    if len(predictions) > 0:
        #return mode
        return stats.mode(predictions,axis=None).mode[0]
    else:
        return None

def predict(image_name,templates,tw=44,th=28,min_w=24,debug=False):
    image = data.imread(image_name)
    image = image[10:80,10:80]
    image = gaussian_filter(image, 1)
    
    
    bb = best_box(image,limit=TWO_DIGIT_LIMIT,debug=debug)
    
    if bb is None:
        return None
        
    x, y, w, h = bb
    
    is_two_digits = w >= TWO_DIGIT_LIMIT
    
    if is_two_digits:
        tw = 44
    else:
        tw = 36 
    
    tb = get_target_bounding_box(bb,tw,th)
    
    #assert tb[2] <= tw, "problem with tw {0}:{1}".format(tb[2],tw)
    #assert tb[3] <= th, "problem with th {0}:{1}".format(tb[3],th)
    
    if debug: print "target box",tb,"is_two_digits",is_two_digits

    if tb is not None:
        x, y, w, h = tb
        x1 = x
        x2 = x + w - max(tw/2,min_w)
        w1 = max(tw/2,min_w)
        w2 = max(tw/2,min_w)

        selection = image[y:y+h,x:x+w]
        selection = rgb2gray(selection)
        prediction = match_digit(selection,templates,min_w=min_w,is_two_digits=is_two_digits,debug=debug)
        
        if is_two_digits and prediction[0] is not None and prediction[1] is not None:
            prediction = prediction[0][1] * 10 + prediction[1][1]
        elif not is_two_digits and prediction[0] is not None:
            prediction = prediction[0][1]
        else:
            prediction = None

    return prediction
