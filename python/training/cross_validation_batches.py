from __future__ import print_function
import sys

sys.path += ["..","."]


import utils
import os
import os.path
import random
import sklearn.cross_validation

import convnet

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
        "../training/1.11",
        "../training/1.12",
        "../training/1.13",
        "../training/1.14",
        "../training/1.15",
        "../training/1.16",
        "../training/1.35",
    ]
    images = set_cross_validation(folders)


    image_list = []
    for i,ims in enumerate(images):
        image_list += ims
        
    print ('total images',len(image_list))
    
    #input_size = 288*384
    w = 384
    h = 288
    
    #h1 = 512
    #h2 = 128
    #h3 = None
    #nb_classes = 2
    
    batch_size = 200
    nb_epoch = 100
    
    print("batch_size",batch_size,"nb_epoch",nb_epoch)
    
    
    kf = sklearn.cross_validation.KFold(len(image_list),n_folds=10,shuffle=True,random_state=1634120)
    
    for i,(train_index, test_index) in enumerate(kf):
        print(train_index)
        print(test_index)
        training_set = [image_list[x] for x in train_index]
        testing_set = [image_list[x] for x in test_index]
    
        print("processing k-fold",i,'training_set',len(training_set),'testing_set',len(testing_set))
        
        #first god/bad classifier
        training_set = [(x,0 if y == 0 else 1) for x,y in training_set]
        testing_set = [(x,0 if y == 0 else 1) for x,y in testing_set]


        random.shuffle(training_set)
        random.shuffle(testing_set)

        model = convnet.make_binary_model((3,h,w))

        X_test,y_test   = utils.make_X_y_convnet_opencv(testing_set)

        for k in xrange(0,len(training_set),batch_size):
            batch_start = k
            batch_end = min(k+batch_size,len(training_set))
        
            X_train,y_train = utils.make_X_y_convnet_opencv(training_set[batch_start:batch_end])
        
            train_score = model.train_on_batch(X_train, y_train)        
        
            test_score = test_on_batch(X_test,y_test)
        
        
        #convnet.save_model(model,"model-binary-cv-{0}.json".format(i),"model-binary-cv-{0}.h5".format(i))        
