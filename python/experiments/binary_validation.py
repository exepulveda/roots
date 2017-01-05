from __future__ import print_function
import sys

sys.path += ["..","."]


import numpy as np
import os
import os.path
import random
import cv2

import prediction
import utils

from config import configuration

def set_cross_validation(folders):
    '''Define a list with all videos. Folders contains a list of paths
    to each video. Each folder has a folder bads and oks. Inside oks folder with frame images
    '''
    images = []
    for images_path in folders:
        #bads images
        filenames = []
        utils.expand_folder(os.path.join(images_path,"rejected"),filenames)
        bads_filenames = [(x,0) for x in filenames]
        #oksimages
        oks_filenames = []
        for i in range(6,55):
            #making folder name
            filenames = []
            utils.expand_folder(os.path.join(images_path,"windows","frame-{0}".format(i)),filenames)
            oks_filenames += [(x,i) for x in filenames]
            
        
        images += [bads_filenames + oks_filenames] 
        
    return images
    
def stats(n,tp,tn,fp,fn,full=False):
    assert(n == (tp + tn + fp + fn))

    if full:
        print("total images:",n)
        print("true positive:",tp, (float(tp)/n) * 100)
        print("true negative:",tn, (float(tn)/n) * 100)
        print("false positive:",fp, (float(fp)/n) * 100)
        print("false negative:",fn, (float(fn)/n) * 100)
    print("accuracy on {0} images: {1}".format(n,float(tp+tn)/float(n) * 100.0))
    

if __name__ == "__main__":
    '''
    '''
    
    folders = [
        '/media/esepulveda/Elements/4-training/1.11',
        '/media/esepulveda/Elements/4-training/1.12',
        '/media/esepulveda/Elements/4-training/1.13',
        '/media/esepulveda/Elements/4-training/1.14',
        '/media/esepulveda/Elements/4-training/1.15',
        '/media/esepulveda/Elements/4-training/1.16',
        '/media/esepulveda/Elements/4-training/1.33',
        '/media/esepulveda/Elements/4-training/1.33-2',
        '/media/esepulveda/Elements/4-training/1.33-3',
        '/media/esepulveda/Elements/4-training/1.33-4',
        '/media/esepulveda/Elements/4-training/1.33-5',
        '/media/esepulveda/Elements/4-training/1.33-6',
        '/media/esepulveda/Elements/4-training/1.33-7',
        '/media/esepulveda/Elements/4-training/1.35',   
    ]
    images = set_cross_validation(folders)


    image_list = []
    for i,ims in enumerate(images):
        image_list += ims

    n = len(image_list)
        
    #load binary_classifier
    binary_model = utils.load_model(configuration.model.classifier + ".json",configuration.model.classifier + ".h5")

        
    target_size = (configuration.input.binary_width,configuration.input.binary_height)

    fn = 0
    fp = 0
    tn = 0
    tp = 0

    for i,(image_name,tag) in enumerate(image_list):
        #load the image
        image = utils.load_image_opencv(image_name)
        image = cv2.resize(image,target_size) 
        image = np.moveaxis(image,-1,0)
        image = image[np.newaxis,:]
        #print(image.shape)
        
        pred = prediction.predict_accepted_rejected(image,binary_model,configuration)
        
        if pred == prediction.REJECTED:
            if tag != 0: #rejected but was accepted --> false negative
                fn += 1
            else:
                tp += 1 #rejected but was rejected --> true positive
        else:
            if tag != 0: #accepted but was accepted --> true negative
                tn += 1
            else: #accepted but was rejected --> false positive
                fp += 1
                
        if i >0 and i % 100 == 0:
            stats(i+1,tp,tn,fp,fn)
                
    
    stats(n,tp,tn,fp,fn)
    
    
