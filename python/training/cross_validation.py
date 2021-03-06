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

if __name__ == "__main__":
    '''
    '''

    training_folder = '/media/esepulveda/Elements/4-training'
    
    folders = [
        '1.11',
        '1.12',
        '1.13',
        '1.14',
        '1.15',
        '1.16',
        '1.33',
        '1.33-2',
        '1.33-3',
        '1.33-4',
        '1.33-5',
        '1.33-6',
        '1.33-7',
        '1.35',   
        ]

    images = set_cross_validation([os.path.join(training_folder,x) for x in folders])

    image_list = []
    for i,ims in enumerate(images):
        image_list += ims
        
    print("total number of images:",len(image_list) )
    #input_size = 288*384
    ow = 384
    oh = 288
    
    tw = ow/4
    th = oh/4

    batch_size = 100
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
        
        print("loading images")
        X_train,y_train = utils.make_X_y_convnet_opencv(training_set,target_size=(th,tw))
        X_test,y_test   = utils.make_X_y_convnet_opencv(testing_set,target_size=(th,tw))
            
        model = convnet.make_binary_model_full((3,th,tw))
        score = convnet.train(model, X_train,X_test,y_train,y_test,1,batch_size,nb_epoch)
        print('CV,', i, ",training size,",len(training_set),",testing size,",len(testing_set),',Test score,', score[0],',Test accuracy,', score[1])
        
        convnet.save_model(model,"model-binary-cv-{0}.json".format(i),"model-binary-cv-{0}.h5".format(i))        
