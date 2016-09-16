'''This script performs a cross validation of window classifier
The model is a convolutional neural network with one convolutional layer and one full connected layer
'''
from __future__ import print_function

import sys

sys.path += ["..","."]

import utils
import os
import os.path
import random
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.cross_validation
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

import convnet
import config

from config import configuration

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
            oks_filenames += [(x,i-6) for x in filenames]
            
        
        images += [oks_filenames] 
        
    return images

def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]
            
class SaverCallback(Callback):
    def __init__(self,cv_number):
        Callback.__init__(self)
        self.cv_number = cv_number
        
    def on_epoch_end(self, epoch, logs={}):
        print("Saving model for this epoch",epoch)
        model_filename = "mode_defs_{0}_{1}.json".format(self.cv_number,epoch)
        model_weights_filename = "mode_weights_{0}_{1}.h5".format(self.cv_number,epoch)
        convnet.save_model(self.model,model_filename,model_weights_filename)

if __name__ == "__main__":
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
        
    print ('total images',len(image_list))
    
    kf = sklearn.cross_validation.KFold(len(image_list),n_folds=5,shuffle=True,random_state=1634120)
    
    nb_classes = configuration.window.end-configuration.window.start+1 #49
    
    batch_size = 100
    nb_epoch = 50
    
    w = 384
    h = 288
    
    offset_w = configuration.window.offset_width
    offset_h = configuration.window.offset_height
    
    target_w = configuration.window.image_width
    target_h = configuration.window.image_height    
    
    #generator = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2)

    #input_size = target_w*target_h
    
    for i,(train_index, test_index) in enumerate(kf):
        #print(train_index)
        #print(test_index)
        training_set = [image_list[x] for x in train_index]
        testing_set = [image_list[x] for x in test_index]
    
        print("processing k-fold",i,'training_set',len(training_set),'testing_set',len(testing_set))
        
        #first god/bad classifier
        #training_set = [(x,0 if y == 0 else 1) for x,y in training_set]
        #testing_set = [(x,0 if y == 0 else 1) for x,y in testing_set]

        random.shuffle(training_set)
        random.shuffle(testing_set)
        
        print("loading images")
        X_train,y_train = utils.make_X_y_convnet_opencv(training_set,offset={"h":offset_h,"w":offset_w},size={"h":target_h,"w":target_w})
        X_test,y_test   = utils.make_X_y_convnet_opencv(testing_set ,offset={"h":offset_h,"w":offset_w},size={"h":target_h,"w":target_w})

        modelcheckpoint = SaverCallback(i+1)


        #training_set = training_set[:5000]
        #testing_set = testing_set[:1000]


        #make_X_y_convnet(images,w,h,offset=None,size=None)

        
        #model = convnet.make_model_1(3,target_w,target_h,nb_filters = 16,nb_conv = 10,nb_classes=nb_classes,dropout=0.5)
        print("making model")
        model = convnet.make_window_model((3,target_h,target_w),nb_classes=nb_classes)
        
        score = convnet.train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch,callbacks=[modelcheckpoint],generator=None)
        #print('After training at:', i, 'Test score:', score[0], 'Test accuracy:', score[1])
        
        #score = convnet.test(model,max_value,mean_value,X_test,y_test,nb_classes)
        
        print('Cross Validation result at:', i, 'Test score:', score[0], 'Test accuracy:', score[1])


        convnet.save_model(model,"model-convnet-{0}.json".format(i),"model-convnet-{0}.h5".format(i))       
        #print("mean",mean_value,"max",max_value)
        
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
