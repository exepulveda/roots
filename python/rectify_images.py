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
import utils

from config import configuration
from rectify import rectify



parser = argparse.ArgumentParser(description="Rectify distortioned window images")
parser.add_argument('-i','--input', type=str,help='input folder name',required=True)
parser.add_argument('-o','--output', type=str,help='output folder name',required=True)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)
parser.add_argument('--force', help='restart from extraction',required=False,default=False,action='store_true')
parser.add_argument('--only', help='list of windows to rectify',required=False,default=[],type=int,nargs='*')

def rectification(folder, outputfolder, configuration,windows=[]):
    # delete outputfolder
    if os.path.exists(outputfolder):
        shutil.rmtree(outputfolder)

    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    image_list = []
    utils.expand_folder(folder, image_list)
    if len(image_list) == 0:
        logging.info("STEP 5: there are not windows to rectify...")
        return
        
    #seed number
    np.random.seed(1634120)
    
    n = len(image_list)

    for k,image_name in enumerate(image_list):
        #extract window number from name
        filename, extension = os.path.splitext(os.path.basename(image_name))
        #filename = "frame-WINDOWS"
        windows_in_image = int(filename.split("-")[1])
        
        if len(windows) == 0  or windows_in_image in windows:
            #print (image_name,windows_in_image)

            im = cv2.imread(image_name)
            rectified, circles, matches = rectify(im,configuration.rectify.iterations)
            
            h, w, colors = im.shape
            # check if resize is needed
            if configuration.rectify.image_width != w or configuration.rectify.image_height != h:
                rectified = cv2.resize(rectified, (configuration.rectify.image_width, configuration.rectify.image_height))

            cv2.imwrite((os.path.join(outputfolder, "{}{}".format(filename, extension))), rectified)

        utils.printProgressBar(k,n)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Processing folder: %s",args.input)
    if not os.path.exists(args.input):
        raise Exception("Input folder does not exist")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    logging.info("Processing folder: %s",args.input)
    rectification(args.input, args.output, configuration,windows=args.only)
    
