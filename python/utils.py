import csv
import os
import os.path
import sklearn.cross_validation
import matplotlib.image as mpimg
import numpy as np
from sklearn import tree
import sklearn.metrics
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors
from PIL import Image

def make_X_y(images,w,h):
    n = len(images)
    X = np.empty((n,w*h))
    y = np.empty(n,dtype=np.int32)

    #built training X and y
    for i,(image_filename,tag) in enumerate(images):
        #print image_filename,tag
        image = Image.open(image_filename).convert('L')
        image = np.array(image) / 255.0

        X[i,:] = image.flatten()
        y[i] = tag

    return X,y

def make_X(images,w,h):
    n = len(images)
    X = np.empty((n,w*h))

    #built training X and y
    for i,image_filename in enumerate(images):
        #print image_filename,tag
        image = Image.open(image_filename).convert('L')
        image = np.array(image) / 255.0

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
