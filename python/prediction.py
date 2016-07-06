from __future__ import print_function

import theano

#theano.config.mode = 'FAST_COMPILE'
#theano.config.linker = 'py'

import utils
import os
import os.path
import random
import json
import numpy as np

from config import configuration

ACCEPTED = 1
REJECTED = 0


def predict_accepted_rejected(image,model,configuration):
    w = configuration.input.image_with
    h = configuration.input.image_height

    binary_mean_image = configuration.model.classifier_mean
    binary_max_image = configuration.model.classifier_max
    
    X = utils.load_image(image,w,h,offset=[0,0],size=[w,h])
        
    X = (X-binary_mean_image)/binary_max_image
        
    X = X[np.newaxis,:]

    ret = model.predict_classes(X, batch_size=1, verbose=0)
    
    return ret[0]

def predict_window(image,model,configuration):
    w = configuration.input.image_with
    h = configuration.input.image_height

    offset_w = configuration.window.offset_with
    offset_h = configuration.window.offset_height
        
    target_w = configuration.window.image_with
    target_h = configuration.window.image_height
    
    window_mean_image = configuration.model.window_mean
    window_max_image = configuration.model.window_max
        
    X = utils.load_image_convnet(image,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
        
    X = (X-window_mean_image)/window_max_image
        
    ret = model.predict_classes(X, batch_size=1, verbose=0)
        
    return ret[0] + configuration.window.start
        
