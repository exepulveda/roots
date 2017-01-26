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

import config
import prediction
import utils
import bound

from config import configuration
from utils import extract_frames_from_video
from restore import select_and_restore
from fixer import fix_prediction
from rectify import rectify

parser = argparse.ArgumentParser(description='Performe image recitifcation of images')
parser.add_argument('-l','--listfile', type=str,help='the filename of list of all videos to process',required=True)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)

def rectification(video, video_status, configuration,windows=[]):
    selected_folder = os.path.join(video_folder, "selected")
    rectified_folder = os.path.join(video_folder, "rectified")

    # target_w = configuration.window.image_width

    if video_status.get("window_rectification", False) and len(windows) == 0:
        logging.info("STEP 5: rectification skipped...")
        return

    logging.info("STEP 5: rectifying selected images...")

    # delete rectified_folder
    if os.path.exists(rectified_folder) and len(windows) == 0:
        shutil.rmtree(rectified_folder)

    if not os.path.exists(rectified_folder):
        os.mkdir(rectified_folder)

    image_list = []
    utils.expand_folder(selected_folder, image_list)

    if len(image_list) == 0:
        logging.info("STEP 5: there are not windows to rectify...")
        return
        
    #seed number
    np.random.seed(1634120)

    for image_name in image_list:
        #extract window number from name
        filename, extension = os.path.splitext(os.path.basename(image_name))
        #filename = "frame-WINDOWS"
        windows_in_image = int(filename.split("-")[1])
        
        if len(windows) == 0  or windows_in_image in windows:
            print (image_name,windows_in_image)

            im = cv2.imread(image_name)
            rectified, circles, matches = rectify(im)
            
            h, w, colors = im.shape
            # check if resize is needed
            if configuration.rectify.image_width != w or configuration.rectify.image_height != h:
                rectified = cv2.resize(rectified, (configuration.rectify.image_width, configuration.rectify.image_height))

            cv2.imwrite((os.path.join(rectified_folder, "{}-rect{}".format(filename, extension))), rectified)

    video_status["window_rectification"] = True
    config.save_video_status(video_folder, video_status)


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
                        # apply rectification
                        destination = os.path.join(selected_folder,"frame-{0}.{1}".format(i,configuration.extension))
                        # copy best window image
                        select_and_restore(image_list,destination,configuration)
                    else:
                        empty_windows += [i]                

                video_status["window_selection"] = True
                config.save_video_status(video_folder, video_status)

            else:
                logging.info("STEP 4: window selection skipped...")

            # STEP 5: rectification
            rectification(video, video_status, configuration,windows=args.rectifyonly)
        else:
            logging.info("Video [%s] does not exists", video)

