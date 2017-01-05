#!/usr/bin/env python
import os
import os.path
import argparse
import random
import shutil
import logging
import cv2
import numpy as np
import pickle

from config import configuration
import config
from rest.video2images import extract_frames_from_video
import prediction
import utils
from rest.find_best_window import select_and_restore
import classify_template_matching
import bound


parser = argparse.ArgumentParser(description='Performe binary classifiacion of a new video')
parser.add_argument('-l','--listfile', type=str,help='the filename of list of all videos to process',required=True)
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)
parser.add_argument('--force', help='restart from extraction',required=False,default=False,action='store_true')

def frames_extraction(video,video_status,configuration):
    video_folder = config.setup_video_folders(video,reset_folder=args.force)
    allframes_folder = os.path.join(video_folder,"allframes")

    if not video_status.get("extracted",False):
        logging.info("STEP 1: extracting videos...")
        image_list,ntotal = extract_frames_from_video(video,allframes_folder,skip=1,extension=configuration.extension)
        video_status["extracted"] = True
        config.save_video_status(video_folder,video_status)
    else:
        logging.info("STEP 1: extraction skipped, loading extracted videos...")
        #loading allframes
        logging.info(allframes_folder)
        image_list = []
        utils.expand_folder(allframes_folder,image_list)
            
    logging.info("frames extracted: {0}".format(len(image_list)))
    return image_list


def max_distance(ids):
    #lokking for the max consecutive distance. list must be ordered
    n = len(ids)
    
    if n <=1:
        return 0,0
    else:
        max_d = 0
        for i in xrange(1,n):
            #print i,ids[i-1],ids[i],max_d
            if (ids[i] - ids[i-1]) > max_d:
                max_d = ids[i] - ids[i-1]
    
        return max_d,ids[i]

def try_swap(s1,element,s2):
    #calculate max_distance of s1 - element and s2 + element
    s1_without = list(s1)
    s2_with = list(s2)
    
    s1_without.remove(element)
    s2_with.append(element)
    s2_with.sort()
    
    return max_distance(s1_without),max_distance(s2_with)

def rank_for_swap(windows):
    ret = []
    for k,v in windows.iteritems():
        v.sort()
        
        d,i = max_distance(v)
        
        if d > 0:
            ret += [((d,i),k)]
        
        #print k,len(v)
    
    ret.sort()
    
    if len(ret) == 0:
        return None
    
    worst_window = ret[-1][1]
    image_to_move = ret[-1][0][1]
    dw = ret[-1][0][0] #distance
    
    #print "worst_window:",worst_window,"image_to_move",image_to_move
    
    #now calculate were to best
    lw = windows[worst_window]

    ret = []
    for k in xrange(6,55):
        if k != worst_window and k in windows:
            s_with = list(windows[k])
            s_with.append(image_to_move)
            s_with.sort()
            
            d1,i1 = max_distance(windows[k])
            #print k,len(windows[k]),d1,i1
            
            d2,i2 = max_distance(s_with)
            #print k,len(s_with),d2,i2
            
            #quit()
            if d2 > 0:
                ret += [((d2-d1),k,i2,d1,d2)]
            
    ret.sort()
    
    #for a in ret:
    #    print a
        
    #quit()
    
    return (worst_window,image_to_move),(ret[0][1],ret[0][2]),ret[0][0]
    
def improve_prediction(image_list,predictions):
    #print len(image_list),len(predictions)
    windows = {}
    for i,image in enumerate(image_list):
        #(head, tail) = os.path.split(image)
        frame_number = int(image.split(".")[0])

        ret = predictions[i]
        if ret not in windows:
            windows[ret] = []
            
        windows[ret] += [frame_number]
        
    improved_windows = improve_solution_dict(windows)
    
    ret = []
    for k,v in improved_windows.iteritems():
        for i in v:
            ret += [(i,k)]

    ret.sort()
    return ret

def improve_solution_dict(windows):
    
    for k,v in windows.iteritems():
        v.sort()
        
    previous_error = 1000000
    
    i = 0
    while True:
        #print "rank_for_swap",i
        r4s = rank_for_swap(windows)
        if r4s:
            (from_window,from_image),(to_window,to_image),error = r4s
            #print (from_window,from_image),(to_window,to_image),error,previous_error
            
            if i > 1000: break

            if error <= 0:
                #print "swaping"
                #proceed
                l1 = windows[from_window]
                l1.remove(from_image)
                
                l2 = windows[to_window]
                l2.append(from_image)
                
            previous_error = error
        else:
            print "WARNING"
        i = i + 1
        
    return windows
        
if __name__ == "__main__":

    if True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    video = os.path.expanduser("/home/esepulveda/Dropbox/ap-es-projects/input_vm/1.24.AVI")
    logging.info("Processing video: [%s]",video)

    video_folder = config.setup_video_folders(video,reset_folder=False)
    video_status = config.load_video_status(video_folder)
    
    rejected_folder = os.path.join(video_folder,"rejected")
    accepted_folder = os.path.join(video_folder,"accepted")
    allframes_folder = os.path.join(video_folder,"allframes")
    window_folder = os.path.join(video_folder,"windows")
    window_folder2 = os.path.join(video_folder,"windows2")
    selected_folder = os.path.join(video_folder,"selected")            

    all_windows = range(configuration.window.start,configuration.window.end+1)                
    
    templates = bound.load_templates(configuration.model.window_templates)    
            
    image_list = []
    utils.expand_folder(accepted_folder,image_list)    
    n = len(image_list)

    #images dict
    images_dict = {}
    for i,image in enumerate(image_list):
        (head, tail) = os.path.split(image)
        frame_number = int(tail.split(".")[0])
        images_dict[frame_number] = image

    #first solution
    if not os.path.exists("initial_solution.dump"):
        windows = {}
        for i,image in enumerate(image_list):
            (head, tail) = os.path.split(image)
            frame_number = int(tail.split(".")[0])


            ret = bound.predict(image,templates)
            if ret not in windows:
                windows[ret] = []
                
            windows[ret] += [frame_number]
        
        pickle.dump(windows,open("initial_solution.dump","w"))
    else:
        windows  = pickle.load(open("initial_solution.dump"))
    
    frame_numbers = np.load("frame_number.npy")
    errors = np.load("errors.npy")

    print("Starting optimisation")
    ret = improve_solution(windows)

    if not os.path.exists(window_folder2):
        os.mkdir(window_folder2)
    
    for k,v in windows.iteritems():
        destination = os.path.join(window_folder2,"frame-{0}".format(k))
        if not os.path.exists(destination):
            os.mkdir(destination)
        
        for im in v:
            image_name = images_dict[im]
            destination = os.path.join(window_folder2,"frame-{0}".format(k))
            shutil.copy(image_name,destination)
