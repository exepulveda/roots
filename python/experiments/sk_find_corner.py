import cv2

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

import numpy as np


clf = joblib.load("digits_cls.pkl")

imname = "/home/esepulveda/Documents/projects/roots/python/processing/2.43.AVI/selected/frame-24.tiff"
#imname = "/home/esepulveda/Dropbox/Roots videos/Accepted images, video 1.11 - Inverted order/68.tiff"

#im = cv2.imread("/home/esepulveda/Dropbox/Roots videos/Accepted images, video 1.11 - Inverted order/68.tiff")
im = cv2.imread(imname)

im_gray = rgb2gray(im)
#im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)




from skimage.feature import match_template


template = cv2.imread("../models/templates/intersection-1.tiff")
template = rgb2gray(template)

th_initial = threshold_otsu(im_gray)

print th_initial

for th in np.arange(1.0,th_initial,-0.05):
    im_gray_copy = np.array(im_gray[0:150,0:150])
    
    #ret, im_th = cv2.threshold(im_gray_copy.copy(), th, 255, cv2.THRESH_BINARY_INV)
    

    cv2.destroyAllWindows()    
    cv2.imshow("Resulting Image with Rectangular ROIs", im_gray_copy)
    cv2.waitKey()

    result = match_template(im_gray_copy, template)
    
    print th,np.max(result)
