import csv
import os
import os.path
import sklearn.cross_validation
import numpy as np
from sklearn import tree
import sklearn.metrics
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors
from PIL import Image

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
        image = Image.open(image_filename)
        
        if shape is None:
            h = image.height
            w = image.width
            l = image.layers
            shape = (l,w,h)

            X = np.empty((n,l,final_w,final_h))
            y = np.empty(n,dtype=np.int32)
        
        
        im = np.array(image,dtype=np.float32)# / 255.0
        #im = np.array(image)# / 255.0
        
        if size is not None and offset is not None:
            #need to reduce image
            #print image.shape,offset,size
            im = im[offset[1]:(offset[1] + size[1]),offset[0]:(offset[0] + size[0]),:]
            im = np.swapaxes(im,0,1)
            #print image.shape

        for c in range(l):
            X[i,c,:,:] = im[:,:,c]
            
        if i ==0:
            print im[:10,:10,0]
        
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
        #print image_filename,tag
        image = Image.open(image_filename)
        if not all_channes:
            image = image.convert('L')
            
        image = np.array(image)# / 255.0
        
        if size is not None and offset is not None:
            #need to reduce image
            #print image.shape,offset,size
            image = image[offset[0]:(offset[0] + size[0]),offset[1]:(offset[1] + size[1])]
            #print image.shape

        X[i,:] = image.flatten()
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
