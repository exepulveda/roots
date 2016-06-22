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
import utils
import sklearn.cross_validation

image_shape = (288,384,3)

images_path = "/home/esepulveda/Documents/projects/roots/1.11/oks"

#building a tuple with image + frame number
starting_frame = 6
ending_frame = 54

all_files = []
for i in range(starting_frame,ending_frame+1):
    frame_files = []
    frame_folder = os.path.join(images_path,"frame-{0}".format(i))
    print i,frame_folder
    utils.expand_folder(frame_folder,frame_files)
    for a in frame_files:
        all_files += [(a,i)]
    
training_files, testing_files = sklearn.cross_validation.train_test_split(all_files,train_size=0.8,random_state=1634120)
    
X_training,y_training = utils.make_X_y(training_files,288,384,(0,0),(90,90))
X_testing,y_testing = utils.make_X_y(testing_files,288,384,(0,0),(90,90))


n1 = 288*384
nclasses = 2

print "starting training...",len(y_training),len(y_testing)

model = RandomForestClassifier()
model = model.fit(X_training, y_training)

y_pred = model.predict(X_testing)
ret = sklearn.metrics.confusion_matrix(y_testing, y_pred)
print ret

print sklearn.metrics.classification_report(y_testing, y_pred)
    
