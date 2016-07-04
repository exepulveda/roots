'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import utils
import os
import os.path
import random
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.cross_validation

import convnet

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
            utils.expand_folder(os.path.join(images_path,"oks","frame-{0}".format(i)),filenames)
            oks_filenames += [(x,i-6) for x in filenames]
            
        
        images += [oks_filenames] 
        
    return images

def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]

if __name__ == "__main__":
    folders = [
        "../training/1.11",
        "../training/1.12",
        "../training/1.13",
        "../training/1.14",
        "../training/1.15",
        "../training/1.16",
    ]
    images = set_cross_validation(folders)
    

    image_list = []
    for i,ims in enumerate(images):
        image_list += ims
        
    print ('total images',len(image_list))
    
    nb_classes = 54-6+1 #49
    
    batch_size = 200
    nb_epoch = 100
    
    w = 384
    h = 288
    
    offset_w = 10
    offset_h = 10
    
    target_w = w // 4
    target_h = h //4

    input_size = target_w*target_h
    
    training_set = image_list

    random.shuffle(training_set)
    
    print("loading images")
    X_train,y_train = utils.make_X_y_convnet(training_set,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
    X_test,y_test = None,None

    print("making model")
    model = convnet.make_model_4(1,target_w,target_h,nb_filters = 20,nb_pool = 2,nb_conv = 3,nb_classes=nb_classes)
    
    score,max_value,mean_value = convnet.train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch)
    convnet.save_model(model,"final-model-convnet.json","final-convnet.h5")       
    print("mean",mean_value,"max",max_value)
