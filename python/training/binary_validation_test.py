import sys

sys.path += ["..","."]

import numpy as np
import utils
import os
import os.path
import random
import sklearn.cross_validation
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


import skimage.filters
import skimage.color

image_name = "/Users/exequiel/projects/roots/python/processing/1.25.AVI/allframes/1.tiff"

        
from skimage import data
from skimage.filters import threshold_adaptive, threshold_otsu
from skimage.color import rgb2grey

def load_th_image(image_name):
    image = data.load(image_name)
    grey = rgb2grey(image)

    th = threshold_otsu(grey)

    bin = grey >= th

    bin[50:-50,50:-50] = 0
    return bin

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold

folders = [
        '/Users/exequiel/projects/roots/training/1.12/',
    ]

images = set_cross_validation(folders)

image_list = []
for i,ims in enumerate(images):
    image_list += ims

kf = KFold(len(image_list),n_folds=10,shuffle=True,random_state=1634120)
for i,(train_index, test_index) in enumerate(kf):
    training_set = [image_list[x] for x in train_index]
    testing_set = [image_list[x] for x in test_index]

    print("processing k-fold",i,'training_set',len(training_set),'testing_set',len(testing_set))
    
    #first god/bad classifier
    training_set = [(load_th_image(x).flatten(),0 if y == 0 else 1) for x,y in training_set]
    testing_set = [(load_th_image(x).flatten(),0 if y == 0 else 1) for x,y in testing_set]

    x_training, y_training = zip(*training_set)
    x_testing, y_testing = zip(*training_set)

    #model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    model = KNeighborsClassifier()
    model.fit(x_training, y_training)
    score = model.score(x_testing, y_testing)

    print i,score
    
