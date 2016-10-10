from __future__ import print_function
import sys

sys.path += ["..","."]


import numpy as np
import os
import os.path
import random
import cv2
from sklearn.metrics import confusion_matrix,accuracy_score

import prediction
import utils

from config import configuration
import classify_template_matching

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
            folder_name = os.path.join(images_path,"windows","frame-{0}".format(i))
            if os.path.exists(folder_name):
                utils.expand_folder(folder_name,filenames)
                oks_filenames += [(x,i) for x in filenames]
            
        
        images += [oks_filenames] 
        
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
    window_model = utils.load_model(configuration.model.window + ".json",configuration.model.window + ".h5")
    #loading templates
    templates = classify_template_matching.load_templates(configuration)
    all_windows = range(configuration.window.start,configuration.window.end+1)
    true_values = {}
    predictions = {}
    for w in all_windows:
        true_values[w] = 0
        predictions[w] = 0

    target_size = (configuration.input.binary_width,configuration.input.binary_height)

    fn = 0
    fp = 0
    tn = 0
    tp = 0
    
    true_values = [None]*n
    predictions = [None]*n
    predictions1 = [None]*n
    predictions2 = [None]*n

    for i,(image_name,tag) in enumerate(image_list):
        predicted_window_1,prob = prediction.predict_window_opencv(image_name,window_model,configuration)
        predicted_window_2 = classify_template_matching.classify_image(image_name,templates)
        
        if predicted_window_1 == predicted_window_2:
            #both methods predict the same, trust
            predicted_window = predicted_window_1
        else:
            if prob > configuration.model.min_probability or predicted_window_2 is None:
                #if 70% of classifier or not classification using matching, trust on classifier
                predicted_window = predicted_window_1
            else:
                #trust on matching
                predicted_window = predicted_window_2

        #print(image_name,tag,predicted_window,prob,predicted_window_2)

        true_values[i] = tag
        predictions[i] = predicted_window
        predictions1[i] = predicted_window_1
        predictions2[i] = predicted_window_2 if predicted_window_2 is not None else 0

                
        if i>0 and i % 1000 == 0:
            print("accuracy_score: ",accuracy_score(true_values[:i], predictions[:i]))
            print("accuracy_score 1: ",accuracy_score(true_values[:i], predictions1[:i]))
            print("accuracy_score 2: ",accuracy_score(true_values[:i], predictions2[:i]))
            

    if True:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(true_values, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        
