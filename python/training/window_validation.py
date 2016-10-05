from __future__ import print_function
import sys

sys.path += ["..","."]


import utils
import os
import os.path
import random
import sklearn.cross_validation
from config import configuration

from classify_template_matching import classify_image, load_templates

def set_cross_validation(folders):
    '''Define a list with all videos. Folders contains a list of paths
    to each video. Each folder has a folder bads and oks. Inside oks folder with frame images
    '''
    images = []
    for images_path in folders:
        #oksimages
        oks_filenames = []
        for i in range(6,55):
            #making folder name
            filenames = []
            utils.expand_folder(os.path.join(images_path,"windows","frame-{0}".format(i)),filenames)
            oks_filenames += [(x,i) for x in filenames]
            
        
        images += [oks_filenames] 
        
    return images

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

    #load binary_classifier
    templates = load_templates(configuration)

    for folder in folders:
        if os.path.exists(folder):
            images = set_cross_validation([folder])


            image_list = []
            for i,ims in enumerate(images):
                image_list += ims
                
            fn = 0
            fp = 0
            tn = 0
            tp = 0

            for image_name,tag in image_list:
                pred = classify_image(image_name,templates,threshold=0.2,debug=False)
                
                #print (tag,pred,image_name)
                if pred == tag:
                    tp += 1 #rejected but was rejected --> true positive
                else:
                    fp += 1
            n = len(image_list)

            if n>0:
                print("total images:",n,(float(tp)/n) * 100,folder)
            else:
                print("total images:",n)

