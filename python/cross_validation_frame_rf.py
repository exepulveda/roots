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

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import sklearn.metrics

def set_cross_validation(folders):
    '''Define a list with all videos. Folders contains a list of paths
    to each video. Each folder has a folder bads and oks. Inside oks folder with frame images
    '''
    images = []
    for images_path in folders:
        oks_filenames = []
        for i in range(6,55):
            #making folder name
            filenames = []
            utils.expand_folder(os.path.join(images_path,"oks","frame-{0}".format(i)),filenames)
            oks_filenames += [(x,i) for x in filenames]
            
        
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

    
    input_size = 80*90
    nb_classes = 54-6+1 #49
    
    for i,image_list in enumerate(images):
        print("processing k-fold",i)
        training_set = []
        testing_set = []
        
        #set i-th is the validation set
        for k  in range(len(images)):
            if k != i:
                training_set += images[k]
        
        testing_set = image_list
        
        #first god/bad classifier
        #training_set = [(x,0 if y == 0 else 1) for x,y in training_set]
        #testing_set = [(x,0 if y == 0 else 1) for x,y in testing_set]

        random.shuffle(training_set)
        random.shuffle(testing_set)
        
        print("loading images")
        print("training_set size",len(training_set))
        print("testing_set size",len(testing_set))

        #print("training_set size",len(training_set)*(input_size/1.0e9))
        #print("testing_set size",len(testing_set)*(input_size/1.0e9))

        print("making model")
        #model = RandomForestClassifier()
        model = SVC(kernel="linear", C=0.025)


        X_train,y_train = utils.make_X_y(training_set,288,384,offset=[25,20],size=[60,50])
        X_test_all,y_test_all = utils.make_X_y(testing_set,288,384,offset=[25,20],size=[60,50]) 

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test_all)
        ret = sklearn.metrics.confusion_matrix(y_test_all, y_pred)
        print(model.score(X_test_all,y_test_all))
