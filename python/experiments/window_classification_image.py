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
parser.add_argument('--image', type=str,help='the image to classify',required=True)
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


    logging.info("Processing image: {}".format(args.image))

    if not os.path.exists(args.image):
        raise Exception("Image does not exist")
    
    logging.info("STEP 1: Window classification...")
    
    #loading templates
    templates = classify_template_matching.load_templates(configuration)
    
    all_windows = range(configuration.window.start,configuration.window.end+1)

    image_name = args.image
    logging.debug("processing image: {}".format(image_name))
    predicted_window = classify_template_matching.classify_image(image_name,templates,debug=True)
    if predicted_window is None:
        logging.info("image could not be classify as window {}".format(image_name))
    else:
        logging.info("image: {}. Predicted window={}".format(image_name,predicted_window))
