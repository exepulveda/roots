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
import glob

import config
import utils

from config import configuration
from rectify import rectify


format_images = "T{tube}_{window:03d}_{year:04d}.{month:02d}.{day:02d}_{session:03d}.jpg"


parser = argparse.ArgumentParser(description="Rectify distortioned window images")
parser.add_argument('-t','--tube', type=str,help='tube name',required=True)
parser.add_argument('-o','--output', type=str,help='output folder name',required=True)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)
parser.add_argument('--force', help='restart from extraction',required=False,default=False,action='store_true')
parser.add_argument('--only', help='list of windows to rectify',required=False,default=[],type=int,nargs='*')

def get_filename(tube,window,year,month,day,session=0):
    return format_images.format(tube=tube,window=window,year=year,month=month,day=day,session=session)

def rectification(images, tube, outputfolder, configuration,windows=[]):
    # delete outputfolder
    if os.path.exists(outputfolder):
        shutil.rmtree(outputfolder)

    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    #image_list = []
    #utils.expand_folder(folder, image_list)
    #if len(image_list) == 0:
    #    logging.info("STEP 5: there are not windows to rectify...")
    #    return
    
    #./processing/2.36.AVI/selected/frame-28.tiff
    
    #get number of different dates
    dates = set([x[1] for x in images])
    dates = list(dates)
    dates.sort()
    
    sessions = {}
    for k,date in enumerate(dates):
        sessions[date] = (k + 1)
    
    logging.info("Number of sessions found: %d", len(sessions))
    
    #seed number
    np.random.seed(1634120)
    
    n = len(images) 

    for k,(image_name,date) in enumerate(images):
        filename, extension = os.path.splitext(os.path.basename(image_name))
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        
        #extract window number from name
        windows_in_image = int(filename.split("-")[1])
        
        if len(windows) == 0  or windows_in_image in windows:
            #print (image_name,windows_in_image)

            im = cv2.imread(image_name)
            rectified, circles, matches = rectify(im,configuration.rectify.iterations)
            
            h, w, colors = im.shape
            # check if resize is needed
            if configuration.rectify.image_width != w or configuration.rectify.image_height != h:
                rectified = cv2.resize(rectified, (configuration.rectify.image_width, configuration.rectify.image_height))

            out_filename = get_filename(tube,windows_in_image,year,month,day,session=sessions[date])
            
            cv2.imwrite(os.path.join(outputfolder,out_filename), rectified)

        utils.printProgressBar(k,n)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Processing tube: %s",args.tube)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    pathname = os.path.join(configuration.home,"processing","{0}-*.AVI".format(args.tube),"selected","*.tiff")
    logging.info("Scanning images for tube with template: %s",pathname)
    
    ret = glob.glob(pathname)
    
    logging.info("Processing %d images for tube: %s",len(ret),args.tube)
    
    
    home = os.path.join(configuration.home,"processing",args.tube)
    
    list_to_process = []
    for filename in ret:
        pathname,_ = os.path.split(filename)
        
        pathname = pathname[len(home)+1:]
        no_selected = os.path.split(pathname)[0]
        date = no_selected.split(".")[0]
        #folder_name, folder_extension = os.path.splitext(no_selected)
        #print(date, filename)
        list_to_process += [(filename,date)]
        
    
    rectification(list_to_process, args.tube, args.output, configuration,windows=args.only)
    
