import cv2

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

from skimage.filters import threshold_otsu

import numpy as np


clf = joblib.load("digits_cls.pkl")

imname = "/home/esepulveda/Documents/projects/roots/python/processing/2.43.AVI/selected/frame-14.tiff"
#imname = "/home/esepulveda/Dropbox/Roots videos/Accepted images, video 1.11 - Inverted order/68.tiff"

#im = cv2.imread("/home/esepulveda/Dropbox/Roots videos/Accepted images, video 1.11 - Inverted order/68.tiff")
im = cv2.imread(imname)

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)


th_initial = threshold_otsu(im_gray)

for th in np.arange(255,th_initial,-2):
    im_gray_copy = np.array(im_gray[10:150,10:150])
    cv2.destroyAllWindows()    
    ret, im_th = cv2.threshold(im_gray_copy.copy(), th, 255, cv2.THRESH_BINARY_INV)

    #cv2.imshow("Resulting Image", im_th)
    #cv2.waitKey()

    im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    print "th",th,th_initial,"#rects:",len(rects)

    for k,rect in enumerate(rects):
        x,y,w,h  = rect
        # Draw the rectangles
        #if True or (w >= 28 and h >= 28):
        if False or (10 <= w <= 65 and 20 <= h <= 45):
            print "rect[",k,"]",x,y,w,h

            cv2.rectangle(im_gray_copy, (rect[0]-5, rect[1]-5), (rect[0] + rect[2]+5, rect[1] + rect[3]+5), (0, 255, 0), 3)
            # Make the rectangular region around the digit
            leng = 70 #int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[(y-5):(y+h+5), (x-5):(x+w+5)]
            
            print "roi,shape",roi.shape,leng,pt1,pt2
            rw,rh = roi.shape
            
            
            # Resize the image
            if rw >= 28 and rh >= 28:
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
                # Calculate the HOG features
                roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
                nbr = clf.predict_proba(np.array([roi_hog_fd], 'float64'))
                print "porb",nbr
                nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
                print "prediction",nbr
                
                cv2.putText(im_gray_copy, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        
    cv2.imshow("Resulting Image with Rectangular ROIs", im_gray_copy)
    cv2.waitKey()
