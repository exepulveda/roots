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


parser = argparse.ArgumentParser(description="Rectify distortioned window images")
parser.add_argument('-t','--tube', type=str,help='tube name',required=True)
parser.add_argument('-w','--window', type=int,help='window',required=True)
parser.add_argument('-d','--date', type=int,help='date',required=True)
parser.add_argument('-s','--session', type=int,help='session',required=True)
parser.add_argument('-p','--pad', type=int,help='pad',required=False,default=configuration.rootfly.extra_pad)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)


def rectification(tube,window,date,session,pad,configuration):
    rootfly_foldername = get_to_rootfly_foldername()

    year = int(date[0:4])
    month = int(date[4:6])
    day = int(date[6:8])

    filename = configuration.rootfly.to_rectify_single_image_template.format(tube=tube,date=sdate,year=year,window=window)
    
    if not os.path.exists(filename):
        logging.error("filename=[%s] does not exist",filename)
        quit()

    im = cv2.imread(filename)
    rectified, circles, matches = rectify(im,configuration.rectify.iterations,pad=pad)
    
    h, w, colors = im.shape
    # check if resize is needed
    if configuration.rectify.image_width != w or configuration.rectify.image_height != h:
        rectified = cv2.resize(rectified, (configuration.rectify.image_width, configuration.rectify.image_height))

    #check if tube folder exists
    outputfolder = os.path.join(rootfly_foldername,tube)
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    out_filename_template = os.path.join(configuration.rootfly.to_rootfly_path_template,configuration.rootfly.template)
    out_filename = utils.get_rootfly_filename(out_filename_template,tube,window,year,month,day,session)
    
    cv2.imwrite(out_filename, rectified)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    logging.info("Processing tube=[%s], windows=[%d], date=[%d], session=[%d]",args.tube,args.window,args.date,args.session)

    #configuration.rootfly.to_copy_from
    sdate = str(args.date)

    rectification(args.tube,args.window,sdate,args.session,args.pad,configuration)
