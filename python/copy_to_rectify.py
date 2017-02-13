#!/usr/bin/env python

from __future__ import print_function
import os
import os.path
import argparse
import csv
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


parser = argparse.ArgumentParser(description="Copy selected images to rectofication process")
parser.add_argument('-t','--tube', type=str,help='tube name',required=False,default=None)
parser.add_argument('-d','--date', type=str,help='date',required=False,default=None)
parser.add_argument('-l','--list', type=str, help='use a list of tubes and dates',required=False,default=None)
parser.add_argument('-v','--verbose', help='output debug info',default=False,action='store_true',required=False)


def copy_images(tube,date,window_filename, configuration):
    #template
    template = configuration.rootfly.template
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

            out_filename = utils.get_rootfly_filename(template,tube,windows_in_image,year,month,day,session=sessions[date])
            
            cv2.imwrite(os.path.join(outputfolder,out_filename), rectified)

        utils.printProgressBar(k,n)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    #check for argument to be list or single
    if args.list is None and (args.tube is None or args.date is None):
        logging.info("You need to indicate --list or (--tube and --date) arguments")
        quit()
        
    if args.list is None:
        #make a list with tube and date
        to_copy = [(args.tube,args.date)]
    else:
        #read csv
        list_filename = args.list
        #check if args.list exists
        if os.path.exists(list_filename):
            reader = csv.reader(open(list_filename),delimiter=",")
            to_copy = [(row[0],row[1]) for row in reader]
        else:
            logging.info("The file [%s] in --list parameter does not exist",list_filename)
            quit()
            
    n = len(to_copy)

    logging.info("Processing %d tubes+dates",n)

    list_to_process = []
    
    for k,(tube,date) in enumerate(to_copy):
        #configuration.rootfly.to_copy_from
        pathname = configuration.rootfly.to_copy_from.format(tube=tube,date=date)
        
        ret = glob.glob(pathname)

        logging.info("Processing tube: [%s], date: [%s], number of windows: [%d]",tube,date,len(ret))

        
        for filename in ret:
            cols = filename.split("/")
            
            #pathname = pathname[len(home)+1:]
            #no_selected = os.path.split(pathname)[0]
            #date = no_selected.split(".")[0]
            #folder_name, folder_extension = os.path.splitext(no_selected)
            #print(date, filename)
            #list_to_process += [(filename,date)]        
            list_to_process += [(tube,date,cols[-1])]
            
        #copy_images(images, tube, outputfolder, configuration,windows=[])

    n = len(list_to_process)
    logging.info("Number of images to copy: [%d]",n)

    for k,(tube,date,filename) in enumerate(list_to_process):
        #copy window image to destination folder
        
        utils.printProgressBar(k,n)
        
    utils.printProgressBar(n,n)


    quit()  
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
    
