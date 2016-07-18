from __future__ import print_function
import os
import os.path
import argparse
import random
import shutil
import logging

from config import configuration
import config
from rest.video2images import extract_frames_from_video
import prediction
import utils
from rest.find_best_window import select_and_restore
import classify_template_matching

parser = argparse.ArgumentParser(description='Performe window classifiacion')
parser.add_argument('--video', type=str,help='the video binary results path',required=True)
parser.add_argument('--verbose', type=bool,help='output debug info',default=False)
parser.add_argument('--window', type=int,help='process a specific window',required=False,default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    logging.info("Processing video: %s",args.video)
    if configuration.window.start <= args.window <= configuration.window.end:
        logging.info("Processing window: %d",args.window)

    video_folder = config.get_video_folder(args.video)

    if not os.path.exists(video_folder):
        raise Exception("Video does not exist")
    
    window_folder = os.path.join(video_folder,"windows")
    selected_folder = os.path.join(video_folder,"selected")
        
    #deleting window_folder

    logging.info("STEP 1: Loading window images...")

    if configuration.window.start <= args.window <= configuration.window.end:
        all_windows = [args.window]
    else:
        all_windows = range(configuration.window.start,configuration.window.end+1)
        #deleting window_folder
        if os.path.exists(selected_folder):
            shutil.rmtree(selected_folder)
            os.mkdir(selected_folder)
        
    images_ok = {}
    for i in all_windows:
        images_ok[i] = []

        
    logging.info("STEP 2: Selecting/building best window...")
    empty_windows = []
    for i in all_windows:
        logging.debug("processing window [%d]",i)
        
        image_list = []
        utils.expand_folder(os.path.join(window_folder,"frame-{0}".format(i)),image_list)    
        n = len(image_list)        
        
        if n > 0:
            #apply rectification
            destination = os.path.join(selected_folder,"frame-{0}.{1}".format(i,configuration.extension))
            #copy best window image
            select_and_restore(image_list,destination,configuration)
        else:
            empty_windows += [i]
