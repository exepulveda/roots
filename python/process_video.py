from __future__ import print_function
import utils
import os
import os.path
import argparse
import random
import shutil

import mlp
import convnet
from config import configuration
import config
from rest.video2images import extract_frames_from_video
import prediction
from rest.find_best_window import select_and_restore

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

    print("Processing video:",args.path)


    video_folder = config.setup_video_folders(args.path)
    rejected_folder = os.path.join(video_folder,"rejected")
    allframes_folder = os.path.join(video_folder,"allframes")
    accepted_folder = os.path.join(video_folder,"accepted")
    selected_folder = os.path.join(video_folder,"selected")


    image_list,ntotal = extract_frames_from_video(args.path,allframes_folder,skip=1,extension=configuration.extension)
    n = len(image_list)
    
    #parche
    #image_list = image_list[:100]
    #n = len(image_list)
    
    print("From {} frames, {} frames extracted".format(ntotal,len(image_list)))

    #load the models
    binary_model = mlp.load_model(configuration.model.classifier,configuration.model.classifier_weights)        
    window_model = mlp.load_model(configuration.model.window,configuration.model.window_weights)       
     
    all_windows = range(configuration.window.start,configuration.window.end+1)

    frames_ok = {}
    for i in range(configuration.window.start,configuration.window.end+1):
        frames_ok[i] = []

    #try to find each configuration.frame_step frame
    for i in range(0,n,configuration.frame_step):
        image = image_list[i]
        print("processing image",i+1,image)
        if prediction.predict_accepted_rejected(image,binary_model,configuration) == prediction.REJECTED:
            #copy rejected file to rejected folder
            shutil.copy(image,rejected_folder)
        else:
            predicted_window = prediction.predict_window(image,window_model,configuration)
            #print("image ACCPETED:",image,predicted_window)
            if predicted_window in all_windows:
                frames_ok[predicted_window] += [i]
                shutil.copy(image,accepted_folder)

    #maximum images in a window
    #max_images = max([len(v) for k,v in frames_ok.items()])
    #print("The maximum images of windows is:",max_images)
    #looking for windows without images
    m = [k for k,v in frames_ok.items() if len(v) == 0]
    if len(m) > 0:
        print("There are windows without images:",m)
        for w in m:
            #find previous frame
            starting = 0
            for k in range(w-1,configuration.window.start,-1):
                if len(frames_ok[k]) > 0:
                    starting = frames_ok[k][-1] + 1
                    break

            #find next frame
            ending = n
            for k in range(w+1,configuration.window.end,1):
                if len(frames_ok[k]) > 0:
                    ending = frames_ok[k][0] - 1
                    break
            #now we have a bound to find a missing window
            print("trying to find window",w,"from",starting,"to",ending)

            for i in range(starting,ending):
                image = image_list[i]
                print("processing image",i+1,image)
                if prediction.predict_accepted_rejected(image,binary_model,configuration) == prediction.REJECTED:
                    #print("image REJECTED:",image)
                    shutil.copy(image,rejected_folder)
                else:
                    predicted_window = prediction.predict_window(image,window_model,configuration)
                    #print("image ACCPETED:",image,predicted_window)
                    if predicted_window in m:
                        frames_ok[predicted_window] += [i]
                        shutil.copy(image,accepted_folder)



    #select a small number of images
    for i in range(configuration.window.start,configuration.window.end+1):
        if len(frames_ok[i]) > configuration.max_images:
            frames_ok[i] = random.sample(frames_ok[i], configuration.max_images)

    for i in range(configuration.window.start,configuration.window.end+1):
        print("{0} accepted images for window {1}".format(len(frames_ok[i]),i))
        #apply rectification
        image_names = [image_list[k] for k in frames_ok[i]]
        destination = os.path.join(selected_folder,"frame-{0}.{1}".format(i,configuration.extension))
        #copy best window image
        select_and_restore(image_names,destination,configuration)
        

    for i in range(configuration.window.start,configuration.window.end+1):
        if frames_ok[i] == 0:
            print("WARNING: No image could be found for window {0}. Please find manually".format(i))
