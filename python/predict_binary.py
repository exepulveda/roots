from __future__ import print_function
import utils
import os
import os.path
import random
import json

import mlp
import convnet
from config import configuration

if __name__ == "__main__":
    model = mlp.load_model(configuration.model.frames,configuration.model.frames_weights)        

    images_path = "../training/1.16"
    oks_filenames = []
    for i in range(6,55):
        #making folder name
        filenames = []
        utils.expand_folder(os.path.join(images_path,"oks","frame-{0}".format(i)),filenames)
        oks_filenames += [(x,i-6) for x in filenames]

    
    w = configuration.input.image_with
    h = configuration.input.image_height
    
    offset_w = configuration.frame.offset_with
    offset_h = configuration.frame.offset_height
        
    target_w = configuration.frame.image_with
    target_h = configuration.frame.image_height
    
    mean_image = configuration.model.frames_mean
    max_image = configuration.model.frames_max
    #mean 0.652293610732 max 0.181551718176
    #0.651631600121 max 0.181967157483
    for i,(image,tag) in enumerate(oks_filenames):
        #X_train = utils.load_image(image,384,288)
        
        X = utils.load_image_convnet(image,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
        
        X = (X-mean_image)/max_image
        
        
        
        
        #print (model.evaluate(X, batch_size=1, verbose=0)
        
        ret = model.predict_classes(X, batch_size=1, verbose=0)
        
        print(image,ret,tag,X.shape)
        
