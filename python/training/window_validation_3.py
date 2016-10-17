from __future__ import print_function
import sys

sys.path += ["..","."]


import utils
import os
import os.path
import random
import sklearn.cross_validation
from config import configuration

from skimage import data
from bound import load_templates
from bound import predict

from fixer import fix_prediction

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
        #'/Users/exequiel/projects/roots/training/1.12',
        #'/Users/exequiel/projects/roots/training/1.13',
        #'/media/esepulveda/Elements/4-training/1.11',
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

    templates = load_templates(configuration.model.window_templates)

    
    for folder in folders:
        if os.path.exists(folder):
            images = set_cross_validation([folder])


            image_list = []
            for i,ims in enumerate(images):
                image_list += ims

            n = len(image_list)

            ss = []
            for image_name,tag in image_list:
                fname, extension = os.path.splitext( os.path.basename(image_name))
                ss += [(int(fname),image_name,tag)]

            ss.sort()

            fn = 0
            fp = 0
            tn = 0
            tp = 0

            debug=False
            x = []
            y = []

            for fnumber,image_name,tag in ss:
                pred = predict(image_name,templates,debug=False)
                pred =  pred if pred else 0 #fix None values
                x += [fnumber]
                y += [pred]
                if pred == tag:
                    tp += 1 #rejected but was rejected --> true positive
                else:
                    fp += 1

            eff1 = float(tp)/n * 100


            fn = 0
            fp = 0
            tn = 0
            tp = 0

            y_fixed = fix_prediction(x,y)

            for tag,pred in zip(y,y_fixed):
            
                if pred == tag:
                    tp += 1 #rejected but was rejected --> true positive
                else:
                    fp += 1
                    if debug: print("BAD:",image_name,tag,pred)

            eff2 = float(tp)/n * 100

            print("eff pred/fix:",n,eff1,eff2,folder)
                
            
