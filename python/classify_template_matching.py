import cv2
import numpy as np
import os.path
from rest.find_bounding_boxes import find_bounding_boxes

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

  
def find_zone(im):
    expected_size=(20,30)
    boundings = find_bounding_boxes(im, expected_size,.25)
    if len(boundings) > 0:
        x1,y1,w1,h1 = boundings[0]
        x2,y2,w2,h2 = boundings[1]
        
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)        
        
        return min(y1,y2),min(x1,y2),max(y1 + h1,y2 + h2),max(x1 + w1,x2 + w2)
    else:
        return 0,0,im.shape[0],im.shape[1]
    
    
def classify_image(image,templates,threshold=0.2,debug=False):
    im = cv2.imread(image)
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #first try to select a smaller rectangle where to find
    #mode_image = np.mean(im)
    expected_size=(20,30)
    boundings = find_bounding_boxes(im, expected_size,tol=.25,histogram_th=0.95)
    if len(boundings) == 0:
        boundings = find_bounding_boxes(im, expected_size,tol=.25,histogram_th=0.90)    
        
    digits = []
    
    for b in boundings:
        x,y,w,h = b
        if debug:
            print w,h
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)    
    
        if h < 30:
            h = 30
        if w < 20:
            w = 20
    
        #adjusting crop
        yc = y#-2
        hc = h#+2
        xc = x#-2
        wc = w#+2
    
        im1 = im[yc:y+hc,xc:x+wc]
        matches = []
        for digit,template in templates.iteritems():
            try:
                res = cv2.matchTemplate(im1,template,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if debug:
                    print digit,max_val,max_loc,threshold

                if max_val > threshold:
                    #we have found one element
                    matches += [(max_val,digit,x)]
            except:
                print "OPECV ERROR:",digit,template.shape,im1.shape,image
                quit()

        n_matches = len(matches)
        #print "n_matches",n_matches
        if n_matches > 0:
            #select the best one
            matches.sort()
            first_match = matches[-1]
            #delete that part
            max_val,digit,x = first_match
            
            digits += [(digit,x)]
    
    if len(digits) == 1:
        return digits[0][0]
    elif len(digits) == 2:
        if digits[0][1] < digits[1][1]:
            number = digits[0][0] * 10 + digits[1][0]
        else:
            number = digits[1][0] * 10 + digits[0][0]
        return number
    else:
        return None

def classify_image_2(image,templates,threshold=0.4,debug=False):
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
            print digit,max_loc,h,w
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
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-6/805.tiff"
    #x = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-6/182.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-6/952.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-14/1540.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-17/79.tiff" #6
    img = cv2.imread(x)

    boundings = find_bounding_boxes(img,(20,30))
    print boundings

    templates = load_templates(configuration)

    img = cv2.imread(x)
    print classify_image(x,templates,threshold=0.2,debug=True) 
    
    
    quit()
    
    
    for b in boundings:
        x,y,w,h = b
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
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
