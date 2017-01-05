import csv
import os
import os.path
import numpy as np
from PIL import Image
from keras.models import model_from_json

import cv2


def load_image_raw(image_filename):
    image_color = Image.open(image_filename)
    image = image_color.convert('L')
    image = np.array(image,dtype=np.float32)
    image = np.swapaxes(image,0,1)    

    return image

def load_image(image_filename,w,h,offset=None,size=None):
    if size is not None and offset is not None:
        final_w = size[0]
        final_h = size[1]
    else:
        final_w = w
        final_h = h
    
    X = np.empty((final_w*final_h))

    image = Image.open(image_filename).convert('L')
    image = np.array(image,dtype=np.float32)
    
    if size is not None and offset is not None:
        #need to reduce image
        #print image.shape,offset,size
        image = image[offset[1]:(offset[1] + size[1]),offset[0]:(offset[0] + size[0])]
        #print image.shape
    image = np.swapaxes(image,0,1)

    X[:] = image.flatten()

    return X

def load_image_opencv(image_filename,offset=None,size=None):
    image = cv2.imread(image_filename)
    
    #shape is (h,w,channels)
    if size is not None and offset is not None:
        image = image[offset["h"]:(offset["h"] + size["h"]),offset["w"]:(offset["w"] + size["w"]),:]
    
    return image


def load_image_convnet(image_filename,w,h,offset=None,size=None):
    if size is not None and offset is not None:
        final_w = size[0]
        final_h = size[1]
    else:
        final_w = w
        final_h = h
    
    X = np.empty((1,1,final_w,final_h))

    image = Image.open(image_filename).convert('L')
    image = np.array(image,dtype=np.float32)
    
    if size is not None and offset is not None:
        #need to reduce image
        #print image.shape,offset,size
        #image = image[offset[0]:(offset[0] + size[0]),offset[1]:(offset[1] + size[1])]
        image = image[offset[1]:(offset[1] + size[1]),offset[0]:(offset[0] + size[0])]
        image = np.swapaxes(image,0,1)
        #print image.shape

    X[0,0,:,:] = image

    return X
def make_X_y_convnet(images,w,h,offset=None,size=None):
    n = len(images)
    
    if size is not None and offset is not None:
        final_w = size[0]
        final_h = size[1]
    else:
        final_w = w
        final_h = h
    
    X = None
    y = None
    
    shape = None

    #built training X and y
    for i,(image_filename,tag) in enumerate(images):
        #print image_filename,tag
        image = Image.open(image_filename).convert('L')
        
        if shape is None:
            h = image.height
            w = image.width
            l = 1
            shape = (l,w,h)

            X = np.empty((n,l,final_w,final_h))
            y = np.empty(n,dtype=np.int32)
        
        
        im = np.array(image,dtype=np.float32)
        #im = np.array(image)# / 255.0
        
        if size is not None and offset is not None:
            #need to reduce image
            #print image.shape,offset,size
            im = im[offset[1]:(offset[1] + size[1]),offset[0]:(offset[0] + size[0])]
            im = np.swapaxes(im,0,1)
            #print image.shape

        X[i,0,:,:] = im[:,:]
            
        #if i ==0:
        #    print im[:10,:10,0]
        
        y[i] = tag

    return X,y

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


def make_X_y(images,w,h,offset=None,size=None,all_channes=False):
    n = len(images)
    
    if size is not None and offset is not None:
        final_w = size[0]
        final_h = size[1]
    else:
        final_w = w
        final_h = h
    
    X = np.empty((n,final_w*final_h))
    y = np.empty(n,dtype=np.int32)

    #built training X and y
    for i,(image_filename,tag) in enumerate(images):
        X[i,:] = load_image(image_filename,w,h,offset,size)
        y[i] = tag

    return X,y

def make_X(images,w,h):
    n = len(images)
    X = np.empty((n,w*h))

    #built training X and y
    for i,image_filename in enumerate(images):
        #print image_filename,tag
        image = Image.open(image_filename)
        if not all_channes:
            image = image.convert('L')
        image = np.array(image)# / 255.0

        X[i,:] = image.flatten()

    return X

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
