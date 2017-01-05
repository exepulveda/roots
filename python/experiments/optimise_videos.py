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


def evaluateInd(individual):
    #finding valid individuals
    ret = 0
    
    #print(individual)
    #sum errors
    #idx = [(i,w) for i,w in enumerate(individual)]
    
    #ret = np.sum(errors[idx])

    ret2 = 0
    for i,w in enumerate(individual):
        #print(i,w,errors.shape)
        ret2 += errors[i,w]

    #assert ret == ret2,"{0} {1}".format(ret,ret2)

    return ret2,

def evaluateInd2(individual,w1=1.0,w2 = 10.0):
    #finding valid individuals
    ret = 0
    
    #print(individual)
    #sum errors
    n = len(individual)
    windows = {}
    for i,w in enumerate(individual):
        #print(i,w,errors.shape)
        if w < 49:
            ret += errors[i,w]
            
            if w not in windows:
                windows[w] = []
            
            windows[w] += [frame_number[i]]

    #analyse if the group has variance
    ret_var = 0
    for k,v in windows.iteritems():
        ids = np.array(v)
        ids_min = np.min(ids)
        ids_max = np.max(ids)
        
        ids_std = (ids - ids_min) / (ids_max - ids_min) #0 - 1
        
        ret_var += np.mean(ids_std)
    

    hw1 = w1/(w1+w2)
    hw2 = w2/(w1+w2)

    v  = hw1*ret-hw2*ret_var
    #print(ret,ret_var,v)

    return v,

def optimise(n_images):
    import random

    from deap import base
    from deap import creator
    from deap import tools
    from deap import algorithms

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    IND_SIZE=n_images

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint,0,48)
    toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_int, n=IND_SIZE)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxUniform,indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=48, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluateInd2)
    
    pop = toolbox.population(n=300)
    
    hf = tools.HallOfFame(1,similar=np.array_equal)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.4, ngen=500, halloffame=hf,stats=stats, verbose=True)
    
    return hf


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
    selected_folder = os.path.join(video_folder,"selected")            

    all_windows = range(configuration.window.start,configuration.window.end+1)                
    
    templates = bound.load_templates(configuration.model.window_templates)    
            
    image_list = []
    utils.expand_folder(accepted_folder,image_list)    
    n = len(image_list)

    #calculate the error to each window
    if not os.path.exists("errors.npy"):
        frame_number = np.empty(n)
        errors = np.empty((n,49))
        
        for i,image in enumerate(image_list):
            (head, tail) = os.path.split(image)
            frame_number[i] = int(tail.split(".")[0])

            ret = bound.predict_all_window(image,templates)
            
            errors[i,:] = ret
                    
        np.save("errors",errors)
        np.save("frame_number",frame_number)
    else:
        frame_number = np.load("frame_number.npy")
        errors = np.load("errors.npy")
        
    max_probability = np.max(errors)
    print("max_probability",max_probability)
    errors = errors / max_probability
    print("max_probability",np.max(errors))
    

    print("Starting optimisation")
    ret = optimise(n)
    
    for i,image_name in enumerate(image_list):
        predicted_window = ret[0][i]
        if predicted_window in all_windows:
            #images_ok[predicted_window] += [image_name]
            
            destination = os.path.join(window_folder,"frame-{0}".format(predicted_window))
            if not os.path.exists(destination):
                os.mkdir(destination)
            
            shutil.copy(image_name,destination)
