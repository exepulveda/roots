#!/usr/bin/env python
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
from config import get_to_rootfly_foldername
from rectify import rectify


format_images = "T{tube}_{window:03d}_{year:04d}.{month:02d}.{day:02d}_{session:03d}.jpg"


parser = argparse.ArgumentParser(description="Rectify distortioned window images")
parser.add_argument('-t','--tube', type=str,help='tube name',required=False)
parser.add_argument('-s','--last-sesssion', type=int,help='tube name',required=False)
parser.add_argument('-l','--list', type=str, help='use a list of tubes and dates',required=False,default=None)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)


def rectification(to_process, configuration):
    n = len(to_process)
    rootfly_foldername = get_to_rootfly_foldername()
    
    for k,(tube,date,last_session,image_name,filename) in enumerate(to_process):
        #print tube,date,image_name,filename
        
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:8])
        
        #extract window number from name
        frame_part = image_name.split("-")[1]
        windows_in_image = int(frame_part.split(".")[0])
        
        im = cv2.imread(filename)
        rectified, circles, matches = rectify(im,configuration.rectify.iterations,pad=configuration.rootfly.pad)
        
        h, w, colors = im.shape
        # check if resize is needed
        if configuration.rectify.image_width != w or configuration.rectify.image_height != h:
            rectified = cv2.resize(rectified, (configuration.rectify.image_width, configuration.rectify.image_height))

        #check if tube folder exists
        outputfolder = os.path.join(rootfly_foldername,tube)
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        out_filename_template = os.path.join(configuration.rootfly.to_rootfly_path_template,configuration.rootfly.template)
        out_filename = utils.get_rootfly_filename(out_filename_template,tube,windows_in_image,year,month,day,last_session+1)
        
        cv2.imwrite(out_filename, rectified)

        utils.printProgressBar(k,n)

    utils.printProgressBar(n,n)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    #check for argument to be list or single
    if args.list is None and (args.tube is None or args.last_sesssion is None):
        logging.error("You need to indicate --list or (--tube and --last-sesssion) arguments")
        quit()
        
    if args.list is None:
        #make a list with tube and date
        to_copy = [(args.tube,args.last_sesssion)]
    else:
        #read csv
        list_filename = args.list
        #check if args.list exists
        if os.path.exists(list_filename):
            reader = csv.reader(open(list_filename),delimiter=",")
            to_copy = [(row[0],int(row[1])) for row in reader]
        else:
            logging.error("The file [%s] in --list parameter does not exist",list_filename)
            quit()

    n = len(to_copy)

    logging.info("Processing %d tubes+dates",n)

    list_to_process = []
    
    for k,(tube,session) in enumerate(to_copy):
        #configuration.rootfly.to_copy_from
        pathname = configuration.rootfly.to_rectify_images_template.format(tube=tube,date="*",year="*")
        ret = glob.glob(pathname)

        logging.info("Processing tube: [%s], last-session: [%s]",tube,session)

        
        for filename in ret:
            cols = filename.split("/")
            
            #['.', 'to_rectify', '2015', '20150128', '1.16-20150128.AVI', 'frame-47.tiff']
            date = cols[-3]
            year = cols[-4]
            assert date[0:4] == year

            list_to_process += [(tube,date,session,cols[-1],filename)]

    rectification(list_to_process, configuration)
    
