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
from keras.models import model_from_json,model_from_yaml

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

def make_model(input_size,h1,h2=None,h3=None,classes=1,lr=0.01,activation='sigmoid',dropout=0.5):
    model = Sequential()
    model.add(Dense(h1, input_shape=(input_size,),init='lecun_uniform'))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    if h2 is not None:
        model.add(Dense(h2,init='lecun_uniform',activation=activation))
        model.add(Dropout(dropout))
    if h3 is not None:
        model.add(Dense(h3,init='lecun_uniform',activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(classes,activation='softmax' if classes > 1 else 'sigmoid',init='lecun_uniform'))

    model.summary()

    model.compile(loss='categorical_crossentropy' if classes > 1 else 'binary_crossentropy',
                  optimizer=SGD(lr=lr,decay=0.1),
                  metrics=['accuracy'])
                  
    return model

def train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch):
    X_train = X_train.astype('float32')

    X_train /= 255.0
    
    if X_test is not None:
        X_test = X_test.astype('float32')
        X_test /= 255.0
    
    # convert class vectors to binary class matrices
    if nb_classes > 1:
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        if X_test is not None:
            Y_test = np_utils.to_categorical(y_test, nb_classes)
    else:
        Y_train = y_train
        if X_test is not None:
            Y_test = y_test
        
    
    if X_test is not None:
        history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        return score

    else:
        history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1)

        return None

def train_on_batch(model, X_train,X_test,y_train,y_test,nb_classes):
    X_train = X_train.astype('float32')
    X_train /= 255.0
    if X_test is not None:
        X_test = X_test.astype('float32')
        X_test /= 255.0
    #print(X_train.shape[0], 'train samples')
    #print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    if X_test is not None:
        Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    model.train_on_batch(X_train, Y_train)
    
    score = model.test_on_batch(X_test, Y_test)
    
    return score

def test(model, X,y,x_mean,x_max,nb_classes):
    X_test = X.astype('float32')
    X_test -= x_mean
    X_test /= x_max

    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y, nb_classes)
    return model.test_on_batch(X_test, Y_test)


def load_model(model_filename,model_weights_filename):
    print("loading",model_filename,model_weights_filename)
    model = model_from_json(open(model_filename).read())    
    model.load_weights(model_weights_filename)

    return model

def save_model(model,model_filename,model_weights_filename):
    json_string = model.to_json()
    open(model_filename, 'w').write(json_string)
    model.save_weights(model_weights_filename,overwrite=True)
