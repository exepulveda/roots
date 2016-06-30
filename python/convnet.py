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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

import utils
import cross_validation
import os
import os.path
import sklearn

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

def make_model_1(img_width,img_height,nb_filters = 20,nb_pool = 2,nb_conv = 3,nb_classes=10,optimizer=SGD(),dropout=0.25):
    model = Sequential()
    
    # first convolutional layer
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='valid',input_shape=(1,img_width, img_height)))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.summary()    


    # setting sgd optimizer parameters
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    
                  
    return model

def train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # pre-process training and testing data
    max_value = np.max(X_train)
    X_train /= max_value
    X_test /= max_value

    mean_value = np.std(X_train)
    X_train -= mean_value
    X_test -= mean_value

    X_train = X_train.reshape(X_train.shape[0], 1, 100, 100)
    X_test = X_test.reshape(X_test.shape[0], 1, 100, 100)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
                    
    score = model.evaluate(X_test, Y_test, verbose=0)

    return score,max_value,mean_value


def test(model,max_value, mean_value, X,y,nb_classes):
    X_test = X.astype('float32')
    X_test /= max_value
    X_test -= mean_value
    X_test = X_test.reshape(X_test.shape[0], 1, 100, 100)

    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y, nb_classes)
    return model.test_on_batch(X_test, Y_test)


def load_model(model_filename,model_weights_filename):
    model = model_from_json(model_filename)
    model.load_weights(model_weights_filename)

    return model

def save_model(model,model_filename,model_weights_filename):
    json_string = model.to_json()
    open(model_filename, 'w').write(json_string)
    model.save_weights(model_weights_filename)
