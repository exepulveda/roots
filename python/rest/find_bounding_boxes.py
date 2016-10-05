import numpy as np
import cv2

SUBIMAGE_X = 10
SUBIMAGE_Y = 10
SUBIMAGE_W = 100
SUBIMAGE_H = 100

EXPECTED_SIZES = {
        0: (19,30),
        1: (12,30),
        2: (19,30),
        3: (19,30),
        4: (19,30),
        5: (19,30),
        6: (19,30),
        7: (19,30),
        8: (19,30),
        9: (19,30)
    }

def find_bounding_boxes(img,exp_size,tol=.25,histogram_th=0.9,overlap=3,debug=False,show=False):
    '''this function try to find the box around numbers to limit the template matching
    '''

    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # crop
    gray = gray[SUBIMAGE_X:SUBIMAGE_W,SUBIMAGE_Y:SUBIMAGE_H]

    # remove noise
    #gray = cv2.medianBlur(gray,5)

    # sharp
    blur = cv2.GaussianBlur(gray,(0,0),5)
    gray = cv2.addWeighted(gray, 3.5,blur,-2.5,0)

    #cv2.imshow('gray', gray)

    # Find threshold
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist2 = hist/np.sum(hist) #normalized
    th = np.where(np.cumsum(hist2) >= histogram_th)[0][0]

    ret, thresh = cv2.threshold(gray, th-1, 256, cv2.THRESH_BINARY_INV);

    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE,(5,5))
    thresh = cv2.erode(thresh,kernel,1)

    #cv2.imshow('thresh', thresh)

    # Find the contours
    contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    boundings = []
    

    ew, eh = exp_size
    if debug: print "expected size: ",ew,eh
    #min_w = 5
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if show:
            cv2.rectangle(img, (SUBIMAGE_X+x, SUBIMAGE_Y+y), (SUBIMAGE_X+x + w, SUBIMAGE_Y+y + h), (0, 255, 0), 1)


        if debug: print "(x,y,w,h): tol: ",x,y,w,h,(1-tol)*ew,(1+tol)*2*ew,(1-tol)*eh,(1+tol)*eh
        #if (1-tol)*ew < w < (1+tol)*2*ew and (1-tol)*eh < h < (1+tol)*eh:
        if (1-tol)*ew  < w < (1+tol)*ew and (1-tol)*eh < h < (1+tol)*eh:

            if w < (1+tol)*ew:
                if debug: print "w,h ",w,h
                boundings.append((SUBIMAGE_X+x, SUBIMAGE_Y+y, w, h))
            else: #2 numbers
                boundings.append((SUBIMAGE_X+x, SUBIMAGE_Y+y, w/2 + overlap, h))

                x += w/2 -overlap
                boundings.append((SUBIMAGE_X+x, SUBIMAGE_Y+y, w/2 + overlap, h))

    if show:
        cv2.imshow('res', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        

    return boundings

def find_bounding_boxes_filtered(img,exp_sizes,tol=.25,histogram_th=0.9,overlap=3,debug=False,show=False):
    '''this function try to find the box around numbers to limit the template matching
    '''
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # crop
    gray = gray[SUBIMAGE_X:SUBIMAGE_W,SUBIMAGE_Y:SUBIMAGE_H]

    # remove noise
    gray = cv2.medianBlur(gray,5)

    # sharp
    blur = cv2.GaussianBlur(gray,(0,0),5)
    gray = cv2.addWeighted(gray, 3.5,blur,-2.5,0)

    #cv2.imshow('gray', gray)

    # Find threshold
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist2 = hist/np.sum(hist) #normalized
    th = np.where(np.cumsum(hist2) >= histogram_th)[0][0]

    ret, thresh = cv2.threshold(gray, th-1, 256, cv2.THRESH_BINARY_INV);

    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE,(5,5))
    thresh = cv2.erode(thresh,kernel,1)

    cv2.imshow('thresh', thresh)

    # Find the contours
    contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    boundings = []
    

    #min_w = 5
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if show:
            cv2.rectangle(img, (SUBIMAGE_X+x, SUBIMAGE_Y+y), (SUBIMAGE_X+x + w, SUBIMAGE_Y+y + h), (0, 255, 0), 1)

        #check expected sizes
        for i,exp_size in enumerate(exp_sizes):
            ew, eh, case = exp_size
            if debug: print "expected size: ",ew,eh
            #if (1-tol)*ew < w < (1+tol)*2*ew and (1-tol)*eh < h < (1+tol)*eh:
            if (1-tol)*ew  < w < (1+tol)*ew and (1-tol)*eh < h < (1+tol)*eh:
                if debug: print "(x,y,w,h): tol: ",x,y,w,h,(1-tol)*ew,(1+tol)*2*ew,(1-tol)*eh,(1+tol)*eh
                boundings.append((case,SUBIMAGE_X+x, SUBIMAGE_Y+y, w, h))

    if show:
        cv2.imshow('res', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        

    return boundings


if __name__ == "__main__":
    filename = '/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-11/3.tiff'
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-6/182.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-6/805.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-6/952.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-17/79.tiff" 
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.35.AVI/accepted/578.tiff"
    x= "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/accepted/251.tiff"
    
    img = cv2.imread(x)

    expected_size=(20,30)# (w,h)
    boundings = find_bounding_boxes(img, expected_size,tol=.20,histogram_th=0.8)

    for b in boundings:
        x,y,w,h = b
        print b
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
