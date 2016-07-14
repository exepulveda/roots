import cv2
import cv2
import numpy as np

from matplotlib import pyplot as plt

import utils
 
f1 = "/home/esepulveda/Documents/projects/roots/templates/32.tiff"

template_image = cv2.imread(f1)
print template_image.shape
h, w, channels = template_image.shape

image_fn = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/accepted/1719.tiff"
#image_fn = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/accepted/1426.tiff"


methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
  cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
  
files = []
utils.expand_folder("/home/esepulveda/Documents/projects/roots/training/1.35/oks/frame-32",files)  

#files += [f1]
for fn in files:
    img = cv2.imread(fn)


    # Apply template Matching
    res = cv2.matchTemplate(img,template_image,cv2.TM_CCOEFF_NORMED)
    #res = cv2.normalize(res,0.0, 1.0, cv2.NORM_MINMAX)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print min_val, max_val, min_loc, max_loc


    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.suptitle(meth)

    plt.show()

