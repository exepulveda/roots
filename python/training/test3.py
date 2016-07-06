import os
import os.path
import sklearn.cross_validation
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from sklearn import tree
import sklearn.metrics
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors

training_X = np.load("training.X.npy")
training_y = np.load("training.y.npy")
n_training = len(training_y)

testing_X = np.load("testing.X.npy")
testing_y = np.load("testing.y.npy")
n_testing = len(testing_y)

n1 = 288*384
nclasses = 2



training_XT = np.empty((n_training,n1))
testing_XT = np.empty((n_testing,n1))
for i in xrange(n_training):
    training_XT[i,:] = training_X[i].flatten()

for i in xrange(n_testing):
    testing_XT[i,:] = testing_X[i].flatten()

#now train
print "starting training...",n_training,n_testing

models = [
        tree.DecisionTreeClassifier(),
        RandomForestClassifier(),
        #sklearn.svm.SVC(),
        sklearn.neighbors.KNeighborsClassifier(),
        sklearn.neighbors.RadiusNeighborsClassifier(),
    ]


for model in models:
    model = model.fit(training_XT, training_y)

    y_pred = model.predict(testing_XT)
    ret = sklearn.metrics.confusion_matrix(testing_y, y_pred)
    print ret

    print sklearn.metrics.classification_report(testing_y, y_pred)
