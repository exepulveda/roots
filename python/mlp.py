'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import utils
import cross_validation
import os
import os.path
import sklearn


batch_size = 128
nb_classes = 2
nb_epoch = 200

# the data, shuffled and split between train and test sets
def load_data(image_list):
    #generale filenames with bads images
    filenames = []
    bad_filenames = []
    utils.expand_folder(os.path.join(images_path,"bads"),bad_filenames)
    for filename in bad_filenames:
        filenames += [(filename,0)]
    ok_filenames = []
    utils.expand_folder(os.path.join(images_path,"oks"),ok_filenames)
    for filename in ok_filenames:
        filenames += [(filename,1)]

    #split the databse training and testing
    training_files, testing_files = sklearn.cross_validation.train_test_split(filenames,train_size=0.8,random_state=1634120)

    X_training,y_training = utils.make_X_y(training_files,288,384)
    X_testing,y_testing = utils.make_X_y(testing_files,288,384)
    return (X_training, y_training), (X_testing, y_testing)



images_path = "../1.11"
(X_train, y_train), (X_test, y_test) = load_data(images_path)

def make_model(input_size,h1,h2,classes):
    model = Sequential()
    model.add(Dense(h1, input_shape=(input_size,),init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(h2,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
                  
    return model

def train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)

    return score
