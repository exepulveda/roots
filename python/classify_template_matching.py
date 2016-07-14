import cv2
import numpy as np
import os.path

from matplotlib import pyplot as plt

import utils
from config import configuration

def load_templates(configuration):
    ret = {}
    for i in range(10):
        tmp_fn = os.path.join(configuration.model.window_templates,"{0}.tiff".format(i))
        if os.path.exists(tmp_fn):
            ret[i] = cv2.imread(tmp_fn)
        
    return ret

SUBIMAGE_X = 10
SUBIMAGE_Y = 10
SUBIMAGE_W = 100
SUBIMAGE_H = 100    
    
def find_bounding_boxes(img,exp_size,tol=0.25):
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
    hist2 = hist/np.sum(hist)
    th = np.where(np.cumsum(hist2) >= .9)[0][0]

    ret, thresh = cv2.threshold(gray, th-1, 256, cv2.THRESH_BINARY_INV);

    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE,(5,5))
    thresh = cv2.erode(thresh,kernel,1)


    # Find the contours
    contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    boundings = []

    ew, eh = exp_size

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (1-tol)*ew < w < (1+tol)*2*ew and (1-tol)*eh < h < (1+tol)*eh:

            if w< (1+tol)*ew:
                #print "w,h ",w,h
                boundings.append((SUBIMAGE_X+x, SUBIMAGE_Y+y, w, h))
            else: #2 numbers
                boundings.append((SUBIMAGE_X+x, SUBIMAGE_Y+y, w/2, h))

                x += w/2
                boundings.append((SUBIMAGE_X+x, SUBIMAGE_Y+y, w / 2, h))

    return boundings

def find_zone(im):
    return 0,0,im.shape[0],im.shape[1]
    
def classify_image(image,templates,threshold=0.4,debug=False):
    im = cv2.imread(image)
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #first try to select a smaller rectangle where to find
    #mode_image = np.mean(im)
    rect = find_zone(im)
    
    h1,w1,h2,w2 = rect
    
    im = im[h1:h2,w1:w2]

    matches = []
    
    first_match = None
    second_match = None

    #apply all templates
    #print "first pass"
    for digit,template in templates.iteritems():
        res = cv2.matchTemplate(im,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if debug:
            print digit,max_val,max_loc,threshold

        if max_val > threshold:
            #we have found one element
            matches += [(max_val,digit,max_loc,template.shape[0],template.shape[1])]

    n_matches = len(matches)
    #print "n_matches",n_matches
    if n_matches > 0:
        #select the best one
        matches.sort()
        first_match = matches[-1]
        #delete that part
        max_val,digit,max_loc,h,w = first_match
        if debug:
            print digit,max_loc
        im2 = np.copy(im)
        im2[max_loc[1]:max_loc[1] + h,max_loc[0]:max_loc[0] + w] = 0#mode_image
        
        if debug:
            plt.imshow(im2)
            plt.show()
        
        #apply all templates
        secondary_matches = []
        #print "secod pass"
        for digit,template in templates.iteritems():
            res = cv2.matchTemplate(im2,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if debug:
                print "secondpass",digit,max_val,max_loc,threshold

            if max_val > threshold:
                #we have found one element
                secondary_matches += [(max_val,digit,max_loc)]

        n_matches = len(secondary_matches)
        if n_matches > 0:
            #select the best one
            secondary_matches.sort()
            second_match = secondary_matches[-1]

    if first_match is not None:
        if second_match is not None:
            #we have identify 2 digits
            number1 = first_match[1]
            number2 = second_match[1]
            if debug:
                print "numbers",number1,number2,first_match[2][0],second_match[2][0]
            #decide order according position on width
            if first_match[2][0] < second_match[2][0]:
                number = number1 * 10 + number2
            else:
                number = number2 * 10 + number1
                
        else:
            #we have identify 1 digit
            number = first_match[1]
    else:
        #we could not identify some digits
        number = None
        
    return number


if __name__ == "__main__":
    x = "/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-23/01213.jpg"
    x = "/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-53/03421.jpg"
    img = cv2.imread(x)
    boundings = find_bounding_boxes(img,(20,40))
    
    
    
    for b in boundings:
        x,y,w,h = b
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    quit()
    templates = load_templates(configuration)

    
    #check accuracy
    folders = [
        ("/home/esepulveda/Documents/projects/roots/training/1.11/oks/frame-53",53),
        ("/home/esepulveda/Documents/projects/roots/training/1.12/oks/frame-53",53),
        ("/home/esepulveda/Documents/projects/roots/training/1.13/oks/frame-53",53),
        ("/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-53",53),
        ("/home/esepulveda/Documents/projects/roots/training/1.15/oks/frame-53",53),
        ("/home/esepulveda/Documents/projects/roots/training/1.16/oks/frame-53",53),
        ("/home/esepulveda/Documents/projects/roots/training/1.35/oks/frame-53",53),
        
        ("/home/esepulveda/Documents/projects/roots/training/1.11/oks/frame-52",52),
        ("/home/esepulveda/Documents/projects/roots/training/1.12/oks/frame-52",52),
        ("/home/esepulveda/Documents/projects/roots/training/1.13/oks/frame-52",52),
        ("/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-52",52),
        ("/home/esepulveda/Documents/projects/roots/training/1.15/oks/frame-52",52),
        ("/home/esepulveda/Documents/projects/roots/training/1.16/oks/frame-52",52),
        ("/home/esepulveda/Documents/projects/roots/training/1.35/oks/frame-52",52),

        ("/home/esepulveda/Documents/projects/roots/training/1.11/oks/frame-20",20),
        ("/home/esepulveda/Documents/projects/roots/training/1.12/oks/frame-20",20),
        ("/home/esepulveda/Documents/projects/roots/training/1.13/oks/frame-20",20),
        ("/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-20",20),
        ("/home/esepulveda/Documents/projects/roots/training/1.15/oks/frame-20",20),
        ("/home/esepulveda/Documents/projects/roots/training/1.16/oks/frame-20",20),
        ("/home/esepulveda/Documents/projects/roots/training/1.35/oks/frame-20",20),

        ("/home/esepulveda/Documents/projects/roots/training/1.11/oks/frame-30",30),
        ("/home/esepulveda/Documents/projects/roots/training/1.12/oks/frame-30",30),
        ("/home/esepulveda/Documents/projects/roots/training/1.13/oks/frame-30",30),
        ("/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-30",30),
        ("/home/esepulveda/Documents/projects/roots/training/1.15/oks/frame-30",30),
        ("/home/esepulveda/Documents/projects/roots/training/1.16/oks/frame-30",30),
        ("/home/esepulveda/Documents/projects/roots/training/1.35/oks/frame-30",30),

        ("/home/esepulveda/Documents/projects/roots/training/1.11/oks/frame-50",50),
        ("/home/esepulveda/Documents/projects/roots/training/1.12/oks/frame-50",50),
        ("/home/esepulveda/Documents/projects/roots/training/1.13/oks/frame-50",50),
        ("/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-50",50),
        ("/home/esepulveda/Documents/projects/roots/training/1.15/oks/frame-50",50),
        ("/home/esepulveda/Documents/projects/roots/training/1.16/oks/frame-50",50),
        ("/home/esepulveda/Documents/projects/roots/training/1.35/oks/frame-50",50),

        ("/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-18",18),
    ]
    
    debug = not True
    for folder,window in folders:        
        images = []
        utils.expand_folder(folder,images)
        ret = [classify_image(cv2.imread(x),templates,threshold=0.4,debug=debug) for x in images]
        
        fails = np.count_nonzero(np.array(ret) - window)
        print (1.0 - fails/len(ret)) * 100.0,folder
    quit()
    
    
    
    image = cv2.imread(x)
    
    
    print "templates",len(templates)
    
    ret = classify_image(image,templates)
    print(ret)
