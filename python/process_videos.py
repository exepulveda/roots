#!/usr/bin/env python

from __future__ import print_function
import os
import os.path
import argparse
import random
import shutil
import logging
import cv2
import numpy as np

from config import configuration
import config
from rest.video2images import extract_frames_from_video
import prediction
import utils
from rest.find_best_window import select_and_restore
import classify_template_matching


parser = argparse.ArgumentParser(description='Performe binary classifiacion of a new video')
parser.add_argument('-l','--listfile', type=str,help='the filename of list of all videos to process',required=True)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)
parser.add_argument('--force', help='restart from extraction',required=False,default=False,action='store_true')

def frames_extraction(video,video_status,configuration):
    video_folder = config.setup_video_folders(video,reset_folder=args.force)
    allframes_folder = os.path.join(video_folder,"allframes")

    if not video_status.get("extracted",False):
        logging.info("STEP 1: extracting videos...")
        image_list,ntotal = extract_frames_from_video(video,allframes_folder,skip=1,extension=configuration.extension)
        video_status["extracted"] = True
        config.save_video_status(video_folder,video_status)
    else:
        logging.info("STEP 1: extraction skipped, loading extracted videos...")
        #loading allframes
        logging.info(allframes_folder)
        image_list = []
        utils.expand_folder(allframes_folder,image_list)
            
    logging.info("frames extracted: {0}".format(len(image_list)))
    return image_list

def binary_classification(image_list,video,video_status,configuration):
    rejected_folder = os.path.join(video_folder,"rejected")
    accepted_folder = os.path.join(video_folder,"accepted")
    allframes_folder = os.path.join(video_folder,"allframes")

    n = len(image_list)

    if not video_status.get("binary_classification",False):
        logging.info("STEP 2: performing binary classification...")
        #load the models
        binary_model = utils.load_model(configuration.model.classifier + ".json",configuration.model.classifier + ".h5")
         
        #try to find each configuration.frame_step frame
        rejected = 0
        accepted = 0
        
        target_size = (configuration.input.binary_width,configuration.input.binary_height)

        #applying frame_step
        images_selected = []
        for i in range(0,n,configuration.frame_step):
            images_selected += [image_list[i]]
            
        n_selected = len(images_selected)
        batch_size = configuration.batch_size
        batches = len(images_selected) // batch_size + 1
    
        #print (len(images_selected),batch_size,batches)

        for i in range(batches):
            starting = i*batch_size
            ending = min(starting+batch_size,n_selected)


            #print (i,starting,ending)
            
            images_batch = np.empty((ending-starting,3,configuration.input.binary_height,configuration.input.binary_width))

        
            for k in range(starting,ending):
                #load the image
                image_name = images_selected[k]
                image = utils.load_image_opencv(image_name)
                image = cv2.resize(image,target_size) 
                image = np.moveaxis(image,-1,0)
                image = image[np.newaxis,:]

                
                images_batch[k-starting,:,:,:] = image
                
            predictions = prediction.predict_accepted_rejected_batch(images_batch,binary_model,configuration)
            
            for k in range(starting,ending):
                image_name = images_selected[k]
                p = predictions[k-starting]
            
                if p == prediction.REJECTED:
                    logging.debug("image REJECTED: %s",image_name)
                    rejected += 1
                    #copy rejected file to rejected folder
                    shutil.copy(image_name,rejected_folder)
                else:
                    accepted += 1
                    logging.debug("image ACCEPTED: %s",image_name)
                    shutil.copy(image_name,accepted_folder)

        logging.info("STEP 3: report...")
        logging.info("frames in total: {0}".format(len(image_list)))
        logging.info("rejected frames: {0}".format(rejected))
        logging.info("accepted frames: {0}".format(accepted))    

        video_status["binary_classification"] = True
        video_status["window_classification"] = False
        
        config.save_video_status(video_folder,video_status)
    else:
        logging.info("STEP 2: binary classification skipped...")
    

def window_classification(video,video_status,configuration):
    accepted_folder = os.path.join(video_folder,"accepted")
    window_folder = os.path.join(video_folder,"windows")
    
    all_windows = range(configuration.window.start,configuration.window.end+1)                

    w = configuration.input.image_width
    h = configuration.input.image_height

    offset_w = configuration.window.offset_width
    offset_h = configuration.window.offset_height
        
    target_w = configuration.window.image_width
    target_h = configuration.window.image_height

    if not video_status.get("window_classification",False):
        logging.info("STEP 3: performing window classification...")

        image_list = []
        utils.expand_folder(accepted_folder,image_list)    
        n = len(image_list)

        #deleting window_folder
        if os.path.exists(window_folder):
            shutil.rmtree(window_folder)
            
        os.mkdir(window_folder)

        logging.info("STEP 3: images to detect window [%d]...",n)
        #load binary_classifier
        window_model = utils.load_model(configuration.model.window + ".json",configuration.model.window + ".h5")
        #loading templates
        templates = classify_template_matching.load_templates(configuration)

        images_ok = {}
        for i in all_windows:
            images_ok[i] = []
            
        ok_2 = 0
        ok_cnn = 0
        ok_mt = 0

        #CNN classifier
        batch_size = configuration.batch_size
        batches = n // batch_size + 1
    
        #print (len(images_selected),batch_size,batches)
        predicted_window_1 = np.empty(n,dtype=np.int32)
        prob = np.empty(n)

        for i in range(batches):
            starting = i*batch_size
            ending = min(starting+batch_size,n)

            images_batch = np.empty((ending-starting,3,configuration.input.binary_height,configuration.input.binary_width))

            #print (i,starting,ending)
        
            for k in range(starting,ending):
                #load the image
                image_name = image_list[k]
                
                im = utils.load_image_opencv(image_name,offset={"w":offset_w,"h":offset_h},size={"w":target_w,"h":target_h})
                im = np.array(im,dtype=np.float32) / 255.0
                im = np.moveaxis(im,-1,0) #move channels to first axis

                im = im[np.newaxis,:,:,:] #add batch channel
                    
                images_batch[k-starting,:,:,:] = im

            predicted_window_1_batch,prob_batch = prediction.predict_window_opencv_batch(images_batch,window_model,configuration)

            predicted_window_1[starting:ending] = predicted_window_1_batch
            prob[starting:ending] = prob_batch

        #MT
        for k,image_name in enumerate(image_list):
            predicted_window_2 = classify_template_matching.classify_image(image_name,templates)

            if predicted_window_1[k] == predicted_window_2:
                #both methods predict the same, trust
                predicted_window = predicted_window_1[k]
                ok_2 += 1
            else:
                if prob[k] > configuration.model.min_probability or predicted_window_2 is None:
                    #if 70% of classifier or not classification using matching, trust on classifier
                    predicted_window = predicted_window_1[k]
                    ok_cnn += 1
                else:
                    #trust on matching
                    predicted_window = predicted_window_2
                    ok_mt += 1

            if predicted_window in all_windows:
                images_ok[predicted_window] += [image_name]
                
                destination = os.path.join(window_folder,"frame-{0}".format(predicted_window))
                if not os.path.exists(destination):
                    os.mkdir(destination)
                
                shutil.copy(image_name,destination)

        logging.info("STEP 3: report...")
        logging.info("STEP 3: window classification by CNN only: [%d]",ok_cnn)
        logging.info("STEP 3: window classification by MT only: [%d]",ok_mt)
        logging.info("STEP 3: window classification by both: [%d]",ok_2)

        for i in all_windows:
            if len(images_ok[i]) == 0:
                logging.info("No images detected for window {0}".format(i))

        video_status["window_classification"] = True
        video_status["window_selection"] = False
        config.save_video_status(video_folder,video_status)
    else:
        logging.info("STEP 3: window classification skipped...")
    

if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    if not os.path.exists(args.listfile):
        raise Exception("Video list does not exist")

    logging.info("Processing video list: %s",args.listfile)
    
    
    fin = open(args.listfile, "r")
    
    video_list = fin.readlines()

    video_list = [x.rstrip() for x in video_list if len(x.rstrip()) > 0]
    
    for video in video_list:
        video = os.path.expanduser(video)
        if os.path.exists(video):
            logging.info("Processing video: [%s]",video)
        
            video_folder = config.setup_video_folders(video,reset_folder=args.force)
            
            video_status = config.load_video_status(video_folder)
            
            rejected_folder = os.path.join(video_folder,"rejected")
            accepted_folder = os.path.join(video_folder,"accepted")
            allframes_folder = os.path.join(video_folder,"allframes")
            window_folder = os.path.join(video_folder,"windows")
            selected_folder = os.path.join(video_folder,"selected")            

            logging.info("STEP 1: Frames extraction...")
            image_list = frames_extraction(video,video_status,configuration)

            logging.info("STEP 2: Binary classification...")
            binary_classification(image_list,video,video_status,configuration)
            
            all_windows = range(configuration.window.start,configuration.window.end+1)                

            logging.info("STEP 3: Window classification...")
            window_classification(video,video_status,configuration)

            logging.info("STEP 4: Window selection...")
            if not video_status.get("window_selection",False):
                logging.info("STEP 4: selecting/creating the best window...")

                empty_windows = []

                for i in all_windows:
                    image_list = []
                    destination = os.path.join(window_folder,"frame-{0}".format(i))
                    utils.expand_folder(destination,image_list)    
                    
                    logging.info("For window {0}, accepted images: {1}".format(i,len(image_list)))
                    if len(image_list) > 0:
                        #apply rectification
                        destination = os.path.join(selected_folder,"frame-{0}.{1}".format(i,configuration.extension))
                        #copy best window image
                        select_and_restore(image_list,destination,configuration)
                    else:
                        empty_windows += [i]                

                video_status["window_selection"] = True
                config.save_video_status(video_folder,video_status)

            else:
                logging.info("STEP 4: window selection skipped...")
        else:
            logging.info("Video [%s] does not exists",video)
            
