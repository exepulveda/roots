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

from bound import filter_valid_boxes, get_bounding_box,limit_bounding_box,best_box,match_digit,load_templates
from bound import get_target_bounding_box


im_path = "/home/esepulveda/Documents/projects/roots/python/processing/1.25.AVI/accepted/1492.tiff"

image = data.imread(im_path)
image = image[10:100,10:100]
image = gaussian_filter(image, 1)

templates = load_templates("/home/esepulveda/Documents/projects/roots/python/models/templates")

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
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)


bb = best_box(image,debug=True)
print "best box",bb
for k,bb in bb.iteritems():
    x, y, w, h = bb
    
    print k,x, y, w, h

    rect = mpatches.Rectangle((x, y), w, h,fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)


    try:
        tw = 44 if w >= 30 else 24
        th = 28
        min_w = 24
        bb = get_target_bounding_box(bb,tw,th)
        
        print "target box",bb

        if bb is not None:
            x, y, w, h = bb
            print "box",x, y, w, h
            rect = mpatches.Rectangle((x, y), w, h,fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            x1 = x
            x2 = x + w - max(tw/2,min_w)
            w1 = max(tw/2,min_w)
            w2 = max(tw/2,min_w)
            rect = mpatches.Rectangle((x1, y), w1, h,fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            rect = mpatches.Rectangle((x2, y), w2, h,fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            selection = image[y:y+h,x:x+w]
            selection = rgb2gray(selection)
            prediction = match_digit(selection,templates,is_two_digits=True,debug=True)
            print prediction

    except Exception as e:
        print e

plt.show()
