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

image_shape = (288,384,3)

images_path = "/home/esepulveda/Documents/projects/roots/1.11"

#open traning
training_filename = "/home/esepulveda/Documents/projects/roots/databases/training.csv"
reader = csv.reader(open(training_filename),delimiter=' ')
training_files = []
for filename,label in reader:
    training_files += [(os.path.join(images_path,filename),label)]
X_training,y_training = utils.make_X_y(training_files,288,384)
#open testing
testing_filename = "/home/esepulveda/Documents/projects/roots/databases/testing.csv"
reader = csv.reader(open(testing_filename),delimiter=' ')
testing_files = []
for filename,label in reader:
    testing_files += [(os.path.join(images_path,filename),label)]
X_testing,y_testing = utils.make_X_y(testing_files,288,384)

n1 = 288*384
nclasses = 2

#now train
print "starting training...",len(y_training),len(y_testing)

models = [
        RandomForestClassifier(),
    ]


for model in models:
    model = model.fit(X_training, y_training)

    y_pred = model.predict(X_testing)
    ret = sklearn.metrics.confusion_matrix(y_testing, y_pred)
    print ret

    print sklearn.metrics.classification_report(y_testing, y_pred)
    
#classify a real database
image_folder_to_classify = "/home/esepulveda/Documents/projects/roots/1.16"

validation_images = []
utils.expand_folder(image_folder_to_classify,validation_images)
print len(validation_images)
validation_images.sort()

X_prediction = utils.make_X(validation_images,288,384)
y_pred = model.predict(X_prediction)

for image,tag in zip(validation_images,y_pred):
    print image,tag


'''
starting training... 2190 548
[[175   6]
 [ 18 349]]
             precision    recall  f1-score   support

          1       0.91      0.97      0.94       181
          2       0.98      0.95      0.97       367

avg / total       0.96      0.96      0.96       548

'''
