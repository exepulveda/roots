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

parser = argparse.ArgumentParser(description='Performe binary classifiacion of a new video')
parser.add_argument('--video', type=str,help='the video filename',required=True)
parser.add_argument('--verbose', type=bool,help='output debug info',default=False)
#parser.add_argument('--tube', type=int,help='the tube number',required=True)
#parser.add_argument('--date', type=str,help='date',required=False,default="20160630")
#parser.add_argument('--time', type=str,help='time',required=False,default="070000")
#parser.add_argument('--session', type=str,help='session',required=False,default="001")
#parser.add_argument('--skip-binary', type=bool,help='skip binary classification',required=False,default=False)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    if not os.path.exists(args.video):
        raise Exception("Video does not exist")

    logging.info("Processing video:",args.video)


    video_folder = config.setup_video_folders(args.video)
    rejected_folder = os.path.join(video_folder,"rejected")
    accepted_folder = os.path.join(video_folder,"accepted")
    allframes_folder = os.path.join(video_folder,"allframes")


    logging.info("STEP 1: Frames extraction...")
    image_list,ntotal = extract_frames_from_video(args.video,allframes_folder,skip=1,extension=configuration.extension)
    n = len(image_list)
    
    logging.info("{} frames extracted".format(len(image_list)))

    #load the models
    binary_model = utils.load_model(configuration.model.classifier + ".json",configuration.model.classifier + ".h5")
     
    #try to find each configuration.frame_step frame
    logging.info("STEP 2: Binary classification...")
    rejected = 0
    accepted = 0
    for i in range(0,n,configuration.frame_step):
        image_name = image_list[i]
        logging.debug("processing image",i+1,image_name)
        
        #load the image
        image = utils.load_image_opencv(image_name)
        
        if prediction.predict_accepted_rejected(image,binary_model,configuration) == prediction.REJECTED:
            logging.debug("image REJECTED:",image_name)
            rejected += 1
            #copy rejected file to rejected folder
            shutil.copy(image_name,rejected_folder)
        else:
            accepted += 1
            logging.debug("image ACCEPTED:",image_name)
            shutil.copy(image_name,accepted_folder)

    logging.info("STEP 3: report...")
    logging.info("{0} frames in total".format(len(image_list)))
    logging.info("{0} rejected frames".format(rejected))
    logging.info("{0} accepted frames".format(accepted))    
