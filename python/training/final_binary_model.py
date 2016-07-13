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
        "../training/1.11",
        "../training/1.12",
        "../training/1.13",
        "../training/1.14",
        "../training/1.15",
        "../training/1.16",
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
    
    training_set = []
        
    for k  in range(len(images)):
        training_set += images[k]
        
        
        #first god/bad classifier
    training_set = [(x,0 if y == 0 else 1) for x,y in training_set]

    random.shuffle(training_set)
        
    print("loading images")
    print("training_set size",len(training_set))

    X_train,y_train = utils.make_X_y(training_set,384,288)
        
    print("making model")
    model = mlp.make_model(input_size,h1,h2=h2,classes=nb_classes)
    print("training model")
    score,max_image,mean_image = mlp.train(model, X_train,None,y_train,None,nb_classes,batch_size,nb_epoch)
        
    mlp.save_model(model,"model-binary.json","model-binary.h5")        
