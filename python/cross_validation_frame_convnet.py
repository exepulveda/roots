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
    
    #for image,tag in images[0]:
    #    print (image,tag)

    nb_classes = 54-6+1 #49
    
    batch_size = 500
    nb_epoch = 100
    
    w = 384
    h = 288
    
    offset_w = 10
    offset_h = 10
    
    target_w = w // 4
    target_h = h //4

    input_size = target_w*target_h
    
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
        #training_set = [(x,0 if y == 0 else 1) for x,y in training_set]
        #testing_set = [(x,0 if y == 0 else 1) for x,y in testing_set]

        random.shuffle(training_set)
        random.shuffle(testing_set)
        
        print("loading images")
        print("training_set size",len(training_set))
        print("testing_set size",len(testing_set))

        print("training_set size",len(training_set)*(input_size/1.0e9))
        print("testing_set size",len(testing_set)*(input_size/1.0e9))

        print("making model")
        #training_set = training_set[:5000]
        #testing_set = testing_set[:1000]


        #make_X_y_convnet(images,w,h,offset=None,size=None)

        X_train,y_train = utils.make_X_y_convnet(training_set,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
        X_test,y_test = utils.make_X_y_convnet(testing_set,w,h,offset=[offset_w,offset_h],size=[target_w,target_h]) 
        
        #model = convnet.make_model_1(3,target_w,target_h,nb_filters = 16,nb_conv = 10,nb_classes=nb_classes,dropout=0.5)
        model = convnet.make_model_3(3,target_w,target_h,nb_classes)
        
        score,max_value, mean_value = convnet.train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch)
        
        score = convnet.test(model,max_value, mean_value, X_test,y_test,nb_classes)
        
        print('Cross Validation result at:', i, 'Test score:', score[0], 'Test accuracy:', score[1])


        convnet.save_model(model,"model-convnet-{0}.json".format(i),"model-convnet-{0}.h5".format(i))       
        
        
        #for epoch in range(nb_epoch):
            #print("processing epoch",epoch)
            ##process batches
            #for start_index, end_index in make_batches(len(training_set), batch_size):
                ##print("processing batch",start_index, end_index)

                #X_train,y_train = utils.make_X_y(training_set[start_index:end_index],288,384,offset=[0,0],size=[100,100])
                #X_test,y_test = utils.make_X_y(testing_set[start_index:end_index],288,384,offset=[0,0],size=[100,100]) 
                
                
                ##print("training model in this batch")
                #score = mlp.train_on_batch(model, X_train,X_test,y_train,y_test,nb_classes)
                
                ##print('Batch result')
                ##print('Test score:', score[0])
                ##print('Test accuracy:', score[1])

            #score = mlp.test(model, X_test_all,y_test_all,nb_classes)
        
            #print('Cross Validation result at:', i,epoch)
            #print('Test score:', score[0])
            #print('Test accuracy:', score[1])
