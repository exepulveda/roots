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

import mlp

def set_cross_validation(folders):
    '''Define a list with all videos. Folders contains a list of paths
    to each video. Each folder has a folder bads and oks. Inside oks folder with frame images
    '''
    images = []
    for images_path in folders:
        #bads images
        filenames = []
        utils.expand_folder(os.path.join(images_path,"bads"),filenames)
        bads_filenames = [(x,0) for x in filenames]
        #oksimages
        oks_filenames = []
        for i in range(6,55):
            #making folder name
            filenames = []
            utils.expand_folder(os.path.join(images_path,"oks","frame-{0}".format(i)),filenames)
            oks_filenames += [(x,i) for x in filenames]
            
        
        images += [bads_filenames + oks_filenames] 
        
    return images

if __name__ == "__main__":
    folders = [
        "/home/a1634120/roots/training/1.11",
        "/home/a1634120/roots/training/1.12",
        "/home/a1634120/roots/training/1.13",
        "/home/a1634120/roots/training/1.14",
        "/home/a1634120/roots/training/1.15",
        "/home/a1634120/roots/training/1.16",
    ]
    images = set_cross_validation(folders)

    print(len(images))

    input_size = 288*384
    
    h1 = 512
    h2 = 128
    h3 = None
    nb_classes = 2
    
    batch_size = 200
    nb_epoch = 100
    
    print("batch_size",batch_size,"nb_epoch",nb_epoch)
    
    for i,image_list in enumerate(images):
        training_set = []
        testing_set = []
        print("processing k-fold",i)
        
        #set i-th is the validation set
        for k  in range(len(images)):
            if k != i:
                training_set += images[k]
        
        testing_set = image_list
        
        #first god/bad classifier
        training_set = [(x,0 if y == 0 else 1) for x,y in training_set]
        testing_set = [(x,0 if y == 0 else 1) for x,y in testing_set]

        random.shuffle(training_set)
        random.shuffle(testing_set)
        
        print("loading images")
        print("training_set size",len(training_set))
        print("testing_set size",len(testing_set))

        #print("training_set size",len(training_set)*(input_size/1.0e9))
        #print("testing_set size",len(testing_set)*(input_size/1.0e9))

        #training_set = training_set[:1000]
        #testing_set = testing_set[:500]
        
        X_train,y_train = utils.make_X_y(training_set,384,288)
        X_test,y_test = utils.make_X_y(testing_set,384,288) 
        
        model = mlp.make_model(input_size,h1,h2=h2,classes=nb_classes)
        
        score,max_value,mean_value = mlp.train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch)
        print("max_value,mean_value",max_value,mean_value) 
        print('CV,', i, ",training size,",len(training_set),",testing size,",len(testing_set),',Test score,', score[0],',Test accuracy,', score[1])
        
        mlp.save_model(model,"model-binary-cv-{0}.json".format(i),"model-binary-cv-{0}.h5".format(i))        
