#!/usr/bin/env python

from __future__ import print_function
import os
import os.path
import shutil
import argparse
import csv
import shutil
import logging
import glob

import utils

from config import configuration
from config import get_to_rectify_foldername


parser = argparse.ArgumentParser(description="Copy selected images to rectofication process")
parser.add_argument('-t','--tube', type=str,help='tube name',required=False,default=None)
parser.add_argument('-d','--date', type=str,help='date',required=False,default=None)
parser.add_argument('-l','--list', type=str, help='use a list of tubes and dates',required=False,default=None)
parser.add_argument('-v','--verbose', help='output debug info',default=False,action='store_true',required=False)

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

    logging.info("Processing %d tube-date tuples",n)

    list_to_process = []
    
    for k,(tube,date) in enumerate(to_copy):
        pathname = configuration.rootfly.to_copy_from.format(tube=tube,date=date,year=date[0:4])
        
        ret = glob.glob(pathname)

        logging.info("Processing tube: [%s], date: [%s], number of windows: [%d]",tube,date,len(ret))

        for filename in ret:
            cols = filename.split("/")
            list_to_process += [(tube,date,cols[-1],filename)]

    n = len(list_to_process)
    logging.info("Number of images to copy: [%d]",n)

    for k,(tube,date,filename,fullpath) in enumerate(list_to_process):
        #copy window image to destination folder
        pathname = configuration.rootfly.to_copy_from.format(tube=tube,date=date,year=date[0:4])
        #print(k,tube,date,filename,fullpath)
        
        inputfilename = fullpath
        destination = configuration.rootfly.to_rectify_path_template.format(tube=tube,date=date,year=date[0:4])
        #create folder if is needed
        try:
            os.makedirs(destination)
        except:
            pass
            
        shutil.copyfile(inputfilename,os.path.join(destination,filename))
        
        #copy_images(images, tube, date, outputfolder, configuration)
        #origin = images, tube, date, outputfolder, configuration
        
        utils.printProgressBar(k,n)
        
    if n > 0: utils.printProgressBar(n,n)

    logging.info("All images have been copied to: [%s]", get_to_rectify_foldername())
