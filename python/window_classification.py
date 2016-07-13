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

parser = argparse.ArgumentParser(description='Performe window classifiacion')
parser.add_argument('--video', type=str,help='the video binary results path',required=True)
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


    logging.info("Processing video:",args.video)

    video_folder = config.get_video_folder(args.video)

    if not os.path.exists(video_folder):
        raise Exception("Video does not exist")
    
    accepted_folder = os.path.join(video_folder,"accepted")
    window_folder = os.path.join(video_folder,"windows")

    logging.info("STEP 1: Loading accepted videos...")
    
    image_list = []
    utils.expand_folder(accepted_folder,image_list)    
    n = len(image_list)
    
    logging.info("Using {} accepted images".format(n))

    logging.info("STEP 2: Window classification...")
    window_model = utils.load_model(configuration.model.window + ".json",configuration.model.window + ".h5")

    all_windows = range(configuration.window.start,configuration.window.end+1)

    images_ok = {}
    for i in all_windows:
        images_ok[i] = []

    rejected = 0
    accepted = 0
    for i,image_name in enumerate(image_list):
        logging.debug("processing image",i+1,image_name)
        
        predicted_window,prob = prediction.predict_window_opencv(image_name,window_model,configuration)
        if predicted_window is None:
            logging.debug("image could not be classify as window",image_name)
        else:
            logging.debug("image ACCEPTED:",image_name,"window",predicted_window,"with probability",prob)
            if predicted_window in all_windows:
                images_ok[predicted_window] += [i]
                
                destination = os.path.join(window_folder,"frame-{0}".format(i,configuration.extension))
                if not os.path.exists(destination):
                    os.mkdir(destination)
                
                shutil.copy(image_name,destination)

    logging.info("STEP 3: report...")
    for i in all_windows:
        if len(images_ok[i]) == 0:
            logging.info("No images detected for window {0}".format(len(images_ok[i])))
