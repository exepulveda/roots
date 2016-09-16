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
import cv2

from config import configuration

ACCEPTED = 1
REJECTED = 0


def predict_accepted_rejected(image,model,configuration):
    w = configuration.input.image_with
    h = configuration.input.image_height

    #binary_mean_image = configuration.model.classifier_mean
    #binary_max_image = configuration.model.classifier_max

    if isinstance(image, basestring):
        
        X = utils.load_image(image,w,h,offset=[0,0],size=[w,h])
        
        
        if shape is None:
            shape = image.shape

            X = np.empty((n,shape[0],shape[1],shape[2]))
            y = np.empty(n,dtype=np.int32)

        image = np.array(image,dtype=np.float32) / 255.0
        
        X[i,:,:,:] = image[:,:,:]        
    else:
        #image is a raw image opened with opencv
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
        X = gray_image[np.newaxis,np.newaxis,:,:]
        
    #X = (X-binary_mean_image)/binary_max_image
        
    #X = X[np.newaxis,:]

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
        
    if isinstance(image, basestring):
        X = utils.load_image_convnet(image,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
    else:
        X = np.empty((1,1,target_w,target_h))
        X[0,0,:,:] = image[offset_w:(offset_w + target_w),offset_h:(offset_h + target_h)]        
        
    X = (X-window_mean_image)/window_max_image
        
    ret = model.predict_classes(X, batch_size=1, verbose=0)
        
    return ret[0] + configuration.window.start
        
def predict_window_opencv(image,model,configuration,min_prob=0.1):
    w = configuration.input.image_with
    h = configuration.input.image_height

    offset_w = configuration.window.offset_with
    offset_h = configuration.window.offset_height
        
    target_w = configuration.window.image_with
    target_h = configuration.window.image_height
    
    #window_mean_image = configuration.model.window_mean
    #window_max_image = configuration.model.window_max
        
    if isinstance(image, basestring):
        im = utils.load_image_opencv(image,offset={"w":offset_w,"h":offset_h},size={"w":target_w,"h":target_h})
        im = np.array(im,dtype=np.float32) / 255.0
        im = np.moveaxis(im,-1,0) #move channels to first axis

        im = im[np.newaxis,:,:,:] #add batch channel

    else:
        im = np.empty((1,1,target_h,target_w))
        im[0,:,:,:] = image[:,offset_h:(offset_h + target_h),offset_w:(offset_w + target_w)] / 255.0
        
        
    #ret = model.predict_classes(im, batch_size=1, verbose=0)
    
    ret2 = model.predict_proba(im, batch_size=1, verbose=0)
    
    #ret2 has probabilities
    ret = [(p,i) for i,p in enumerate(ret2[0]) if p >= min_prob]
    
    ret.sort()
    
    if len(ret) >= 1:
        return ret[-1][1] + configuration.window.start, ret[-1][0]
    else:
        return None,None
        
