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

from optimise_videos_2 import improve_prediction

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
        #'/media/esepulveda/Elements/4-training/1.11',
        #'/media/esepulveda/Elements/4-training/1.12',
        #'/media/esepulveda/Elements/4-training/1.13',
        #'/media/esepulveda/Elements/4-training/1.14',
        #'/media/esepulveda/Elements/4-training/1.15',
        #'/media/esepulveda/Elements/4-training/1.16',
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

            windows_list = []
            for image_name,tag in image_list:
                (head, tail) = os.path.split(image_name)
                frame_number = int(tail.split(".")[0])
                
                windows_list += [(frame_number,image_name,tag)]  

                
            windows_list.sort()
                
            image_list = [(image_name,tag) for tail,image_name,tag in windows_list]
            
            fn = 0
            fp = 0
            tn = 0
            tp = 0

            debug=False

            #build prediction
            file_names = []
            initial_predictions = []
            for image_name,tag in image_list:
                if debug: print (image_name,os.path.exists(image_name))
                
                (head, tail) = os.path.split(image_name)
                
                file_names += [tail]  
                          
                pred = predict(image_name,templates,debug=False)
                
                initial_predictions += [pred]
                
            predictions = improve_prediction(file_names,initial_predictions)

            for k,(index,pred) in enumerate(predictions):
                if debug: print (k,index,pred,image_list[k])
                
                #pred = predict(image_name,templates,debug=False)
                
                image_name,tag = image_list[k]
                if pred == tag:
                    tp += 1 #rejected but was rejected --> true positive
                else:
                    fp += 1
                    if debug: print("BAD:",image_name,tag,pred)

            n = len(image_list)
            
            if n>0:
                print("total images:",n,(float(tp)/n) * 100,folder)
            else:
                print("total images:",n)
                
            
