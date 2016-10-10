import cv2
import numpy as np
import os.path
from rest.find_bounding_boxes import find_bounding_boxes
from rest.find_bounding_boxes import find_bounding_boxes_filtered
from rest.find_bounding_boxes import EXPECTED_SIZES

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
    
def classify_image(image,templates,threshold=0.2,debug=False,show=False):
    im = cv2.imread(image)
    ret = rank_image(im,templates,debug=debug,show=show)
    
    t = []
    for k,v in ret.iteritems():
        t += [(v,k)]
    
    t.sort()
    
    if debug:
        for v,k in t:
            print v,k
            
    return t[-1][1]
    
def classify_image_old(image,templates,threshold=0.2,debug=False):
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


def rank_image(im,templates,window_range=(6,54),debug=False,show=False):
    digits1,digits2 = find_digits_image(im,templates,debug=debug,show=show)
    #print digits
    
    ret = {}
    #digits is tuple (digit,error,part)
    for w in range(window_range[0],window_range[1]+1):
        if digits1 is None and digits2 is None:
            error = -10.0
        else:
            if w < 10:
                if digits2 is not None:
                    #wrong case, it should be 1
                    error = -10.0
                elif len(digits1) > w:
                    if debug: print "ranking",w,digits1[w][0]
                    error = digits1[w][0]
                else:
                    error = -10.0
            else:
                if digits2 is None:
                    #wrong case, it should be 2
                    error = -10.0
                else:
                    d1,d2 = str(w)
                    d1 = int(d1)
                    d2 = int(d2)
                    if debug: print "ranking",w,d1,d2,digits1[d1],digits2[d2]
                    #d1 it should be in first box
                    error1 = digits1[d1][0]
                    error2 = digits2[d2][0]
                        
                    error = (error1 + error2)/2.0
            
                
        ret[w] = error
            
    return ret

def find_digits_image(im,templates,debug=False,show=False):
    #im = cv2.imread(image)
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #first try to select a smaller rectangle where to find
    #mode_image = np.mean(im)
        
    digits = []

    expected_sizes = [(15,30,True),(19,30,True),(34,30,False),(38,30,False)]

    boundings = find_bounding_boxes_filtered(im,expected_sizes,tol=.25,histogram_th=0.95,debug=debug,show=show)
    boundings.sort()
    
    if debug:
        print "find_digits_image::len(boundings)",len(boundings)

    if len(boundings) == 0:
        boundings = find_bounding_boxes_filtered(im, expected_sizes,tol=.25,histogram_th=0.90,debug=debug,show=show)    
        
    if debug:
        print "len(boundings)",len(boundings)

    if len(boundings) == 0:
        return None,None

    if len(boundings) >= 2:
        #try to join
        #bounding1 = boundings[0]
        #bounding2 = boundings[1]
        
        is_one_digit1,x1,y1,w1,h1 = boundings[0]
        is_one_digit2,x2,y2,w2,h2 = boundings[1]
        
        xend = max(x1+w1,x2+w2)
        yend = max(y1+h1,y2+h2)
        
        
        boundings[0] = (False,min(x1,x2),min(y1,y2),xend - min(x1,x2), yend - min(y1,y2))
        
        
    #we have found boxes, try to 
    bounding = boundings[0]
    is_one_digit,x,y,w,h = bounding
    
    if debug: print "find_digits_image:",is_one_digit,x,y,w,h
    
    h = max(h,30)
    w = max(w,20)

    if debug: print bounding
    
    if is_one_digit: #one number
        im2 = im[y:y+h,x:x+w]
        
        if show:
            cv2.imshow('res', im2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()         
        rank1 = rank_digits_box(im2,templates,debug=debug)
        #rank1.sort()
        digit1 = rank1
        digit2 = None
        
    else:
        im2 = im[y:y+h,x:x+19]
        rank1 = rank_digits_box(im2,templates,debug=debug)
        #ank1.sort()
        digit1 = rank1

        if show:
            cv2.imshow('res', im2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()         

        if debug: print "rank1",rank1

        #get the best digit1
        lcopy = list(rank1)
        lcopy.sort()
        d1 = lcopy[-1][2]
        w1,h1 = EXPECTED_SIZES[d1]

        if debug: print "expected d1",lcopy,d1,w1,h1


        im2 = im[y:y+h,x+w1+1:x+w1+1+19]
        rank2 = rank_digits_box(im2,templates,debug=debug)
        #rank2.sort()
        digit2 = rank2

        if show:
            cv2.imshow('res', im2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()         

        if debug:
            print "rank2",rank2


    return digit1,digit2
    
def rank_digits_box(im,templates,debug=False):
    '''this function rank all digits in an box
    '''
    digits = []

    bh,bw,_ = im.shape

    for digit,template in templates.iteritems():
        ew,eh = EXPECTED_SIZES[digit]
        th,tw,_ = template.shape

        w = max(min(bw,ew),tw)
        h = max(min(bh,eh),th)

        if debug: print w,h,bw,bh
        
        im1 = im[:h,:w]
        
        try:
            res = cv2.matchTemplate(im1,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if debug:
                print digit,max_val,max_loc

            digits += [(max_val,max_loc,digit)]


        except Exception as e:
            print "OPECV ERROR 1:",digit,template.shape,im1.shape,e
            digits += [(-10.0,0,digit)]
    
    return digits
    

if __name__ == "__main__":
    x = "/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-23/01213.jpg"
    x = "/home/esepulveda/Documents/projects/roots/training/1.14/oks/frame-53/03421.jpg"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-6/805.tiff"
    #x = "/home/esepulveda/Documents/projects/roots/python/processing/1.14.AVI/windows/frame-6/182.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-6/952.tiff"
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-14/1540.tiff"

    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-17/660.tiff" #6
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-7/146.tiff" #6
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-17/660.tiff" #6
    
    x = "/home/esepulveda/Documents/projects/roots/python/processing/2.23.AVI/windows/frame-50/1016.tiff" # 11 but 41

    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.24.AVI/windows/frame-50/2326.tiff" # 11 but 41
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.24.AVI/windows/frame-9/298.tiff" # 9 and 9
    x = "/home/esepulveda/Documents/projects/roots/python/processing/1.24.AVI/allframes/780.tiff" # 13

    templates = load_templates(configuration)
    
    ret = classify_image(x,templates,threshold=0.2,debug=True,show=True)
    print ret

    quit()



    
    #find_bounding_boxes(img,exp_size,tol=.25,histogram_th=0.9,overlap=3)
    window = 10
    s1 = EXPECTED_SIZES[1]
    s2 = EXPECTED_SIZES[0]
    es = (s1[0]+s2[0],(s1[1]+s2[1])//2)
    boundings = find_bounding_boxes(img,es,tol=.25,histogram_th=0.99,overlap=3,debug=True,show=True)
    #print boundings
    for b in boundings:
        x,y,w,h = b
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    quit()
    

    #img = cv2.imread(x)
    ret = rank_image(img,templates,debug=True) 
    #ret.sort()
    
    for k,v in ret.iteritems():
        print k,v
        
    
    
    
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
