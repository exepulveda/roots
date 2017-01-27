import sys
import csv
import os
import os.path
import numpy as np

import cv2

def load_image_opencv(image_filename,offset=None,size=None):
    image = cv2.imread(image_filename)
    
    #shape is (h,w,channels)
    if size is not None and offset is not None:
        image = image[offset["h"]:(offset["h"] + size["h"]),offset["w"]:(offset["w"] + size["w"]),:]
    
    return image


def make_X_y_convnet_opencv(images,offset=None,size=None,target_size=None):
    n = len(images)
    
    X = None
    y = None
    
    shape = None

    #built training X and y
    for i,(image_filename,tag) in enumerate(images):
        #print image_filename,tag
        image = load_image_opencv(image_filename,offset=offset,size=size)
        
        if target_size is not None:
            image = cv2.resize(image,target_size) 
        
        #image shape is (h,w,channels) but keras needs (channel,h,w)
        image = np.moveaxis(image,-1,0) #move channels to first axis
        
        if shape is None:
            shape = image.shape

            X = np.empty((n,shape[0],shape[1],shape[2]))
            y = np.empty(n,dtype=np.int32)

        image = np.array(image,dtype=np.float32) / 255.0
        
        X[i,:,:,:] = image[:,:,:]
        y[i] = tag

    return X,y


def expand_folder(path,container):
    for dirpath, dirnames, files in os.walk(path):
        #print dirpath,len(dirnames),len(files)
        #for name in files:
        #    filename = os.path.join(dirpath, name)
        #    print filename
        #    files += [filename]
        for name in files:
            container += [os.path.join(dirpath,name)]

def load_model(model_filename,model_weights_filename):
    from keras.models import model_from_json

    #print("loading",model_filename,model_weights_filename)
    model = model_from_json(open(model_filename).read())    
    model.load_weights(model_weights_filename)

    return model

def save_model(model,model_filename,model_weights_filename):
    json_string = model.to_json()
    open(model_filename, 'w').write(json_string)
    model.save_weights(model_weights_filename,overwrite=True)

def extract_frames_from_video(video_path,frames_path,skip=15,extension="tiff"):
    # Create output dir
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    vidcap = cv2.VideoCapture(video_path)

    image_list = []
    i=1
    k=1

    success = True
    while success:
        success, frame = vidcap.read()

        #print success,i,video_path
        if success:
            if (i % skip) == 0:
                image_filename = os.path.join(frames_path,'{0}.{1}'.format(k,extension))
                cv2.imwrite(image_filename, frame)
                #print "extracting",i,k,image_filename
                image_list += [image_filename]
                k += 1
            i += 1

    vidcap.release()
    return image_list,i

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, bar_length=100, fill = '*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = fill * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    
    #print ('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        sys.stdout.write('\n')
    sys.stdout.flush()
