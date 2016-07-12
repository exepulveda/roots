from __future__ import print_function
import os
import os.path
import argparse
import random
import shutil

from config import configuration
import config
from rest.video2images import extract_frames_from_video
import prediction
import utils
from rest.find_best_window import select_and_restore

parser = argparse.ArgumentParser(description='Process a new video')
parser.add_argument('--path', type=str,help='the video filename',required=True)
parser.add_argument('--tube', type=int,help='the tube number',required=True)
parser.add_argument('--date', type=str,help='date',required=False,default="20160630")
parser.add_argument('--time', type=str,help='time',required=False,default="070000")
parser.add_argument('--session', type=str,help='session',required=False,default="001")
parser.add_argument('--skip-binary', type=bool,help='skip binary classification',required=False,default=False)

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
    window_folder = os.path.join(video_folder,"windows")


    print("STEP 1: Frames extraction...")
    image_list,ntotal = extract_frames_from_video(args.path,allframes_folder,skip=1,extension=configuration.extension)
    n = len(image_list)
    
    #parche
    #image_list = image_list[:100]
    #n = len(image_list)
    
    print("{} frames extracted".format(len(image_list)))

    #load the models
    binary_model = utils.load_model(configuration.model.classifier,configuration.model.classifier_weights)        
    window_model = utils.load_model(configuration.model.window,configuration.model.window_weights)       
     
    all_windows = range(configuration.window.start,configuration.window.end+1)

    images_ok = {}
    for i in range(configuration.window.start,configuration.window.end+1):
        images_ok[i] = []

    #try to find each configuration.frame_step frame
    print("STEP 2: Preliminary Classification...")
    for i in range(0,n,configuration.frame_step):
        image_name = image_list[i]
        print("processing image",i+1,image_name)
        
        #load the image
        image = utils.load_image_raw(image_name)
        
        if prediction.predict_accepted_rejected(image,binary_model,configuration) == prediction.REJECTED:
            print("image REJECTED:",image_name)
            #copy rejected file to rejected folder
            shutil.copy(image_name,rejected_folder)
        else:
            shutil.copy(image_name,accepted_folder)
            predicted_window,prob = prediction.predict_window_opencv(image_name,window_model,configuration)
            if predicted_window is None:
                print("image could not be classify as window",image_name)
            else:
                print("image ACCEPTED:",image_name,"window",predicted_window,"with probability",prob)
                if predicted_window in all_windows:
                    images_ok[predicted_window] += [(i,None)]
                    
                    destination = os.path.join(window_folder,"frame-{0}".format(i,configuration.extension))
                    if not os.path.exists(destination):
                        os.mkdir(destination)
                    
                    shutil.copy(image_name,destination)

    quit()
    #maximum images in a window
    #max_images = max([len(v) for k,v in frames_ok.items()])
    #print("The maximum images of windows is:",max_images)
    #looking for windows without images
    m = [k for k,v in images_ok.items() if len(v) == 0]
    if len(m) > 0:
        print("STEP 2b: Secondary Classification...")
        print("There are windows without images:",m)
        for w in m:
            #find previous frame
            starting = 0
            for k in range(w-1,configuration.window.start,-1):
                if len(images_ok[k]) > 0:
                    starting = images_ok[k][0][-1] + 1
                    break

            #find next frame
            ending = n
            for k in range(w+1,configuration.window.end,1):
                if len(images_ok[k]) > 0:
                    ending = images_ok[k][0][0] - 1
                    break
            #now we have a bound to find a missing window
            print("trying to find window",w,"from",starting,"to",ending)

            for i in range(starting,ending):
                image_name = image_list[i]
                image = utils.load_image_raw(image_name)
                print("processing image",i+1,image_name)
                if prediction.predict_accepted_rejected(image,binary_model,configuration) == prediction.REJECTED:
                    #print("image REJECTED:",image)
                    shutil.copy(image_name,rejected_folder)
                else:
                    predicted_window = prediction.predict_window(image,window_model,configuration)
                    #print("image ACCPETED:",image,predicted_window)
                    if predicted_window in m:
                        images_ok[predicted_window] += [(i,None)]
                        shutil.copy(image_name,accepted_folder)



    #select a small number of images
    print("STEP 3: Reduce number of windows...")
    for i in range(configuration.window.start,configuration.window.end+1):
        if len(images_ok[i]) > configuration.max_images:
            images_ok[i] = random.sample(images_ok[i], configuration.max_images)

    print("STEP 4: Selecting/buildong best window...")
    empty_windows = []
    for i in range(configuration.window.start,configuration.window.end+1):
        print("{0} accepted images for window {1}".format(len(images_ok[i]),i))
        if len(images_ok[i]) > 0:
            #apply rectification
            images = [image_list[k[0]] for k in images_ok[i]] #0 --> index,  1--> image
            destination = os.path.join(selected_folder,"frame-{0}.{1}".format(i,configuration.extension))
            #copy best window image
            select_and_restore(images,destination,configuration)
        else:
            empty_windows += [i]
        

    print("STEP 5: Final report...")
    print("There are",len(images_ok) - len(empty_windows),"window selected from a total of",len(images_ok))
    for i in empty_windows:
        print("WARNING: No image could be found for window {0}. Please find one manually".format(i))
