from __future__ import print_function
import utils
import os
import os.path
import argparse

import mlp
import convnet
from config import configuration
from rest.video2images import extract_frames_from_video

parser = argparse.ArgumentParser(description='Process a new video')
parser.add_argument('--path', type=str,help='the video filename',required=True)
parser.add_argument('--tube', type=int,help='the tube number',required=True)
parser.add_argument('--date', type=str,help='date',required=False,default="20160630")
parser.add_argument('--time', type=str,help='time',required=False,default="070000")
parser.add_argument('--session', type=str,help='session',required=False,default="001")

if __name__ == "__main__":
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        raise Exception("Video does not exist")


    image_list = extract_frames_from_video(args.path,tmp_dir)

    #load the models
    binary_model = mlp.load_model(configuration.model.frames,configuration.model.frames_weights)        
    window_model = mlp.load_model(configuration.model.frames,configuration.model.frames_weights)        
   
    w = configuration.input.image_with
    h = configuration.input.image_height
    
    offset_w = configuration.frame.offset_with
    offset_h = configuration.frame.offset_height
        
    target_w = configuration.frame.image_with
    target_h = configuration.frame.image_height
    
    mean_image = configuration.model.frames_mean
    max_image = configuration.model.frames_max
    
    frames_ok = {}
    for i in range(6,56):
        frames_ok[i] = 0
    
    for i,image in enumerate(image_list):
        #X_train = utils.load_image(image,384,288)
        #classify acepted/rejected
        X = utils.load_image(image,w,h)
        ret = model.predict(X, batch_size=1, verbose=0)
        if ret == 0:
            #rejected
            pass
        else:
            #classify window
            X = utils.load_image_convnet(image,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
            X = (X-mean_image)/max_image
            ret = window_model.predict_classes(X, batch_size=1, verbose=0) + configuration.frame.offset_class
            #check if there is a previous frame
            k = frames_ok[ret]
            
            frames_ok[i] += 1
            
            #copy tmp_filename to folder video/frame-{}/1.jpg
            
        
