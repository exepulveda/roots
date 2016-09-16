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
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

import utils
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

def make_model_2(img_channels, img_rows, img_cols,nb_classes):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512),init='lecun_uniform')
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes),init='lecun_uniform')
    model.add(Activation('softmax'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    return model

def make_model_3(img_channels, img_rows, img_cols,nb_classes):
    model = Sequential()

    model.add(Convolution2D(16, 5, 5, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 4, 4, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(8, 4, 4, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(256,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes,init='lecun_uniform'))
    model.add(Activation('softmax'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    return model

def make_model_1(channels,img_width,img_height,nb_filters = 20,nb_pool = 2,nb_conv = 3,nb_classes=10,optimizer=SGD(lr=0.01),dropout=0.5):
    model = Sequential()
    
    # first convolutional layer
    model.add(Convolution2D(32,7,7,border_mode='same',input_shape=(channels,img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    #model.add(Convolution2D(256, 4, 4))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    #model.add(Convolution2D(64, 4, 4))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.summary()    


    # setting sgd optimizer parameters
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    
    return model

def make_window_model(input_shape,nb_classes,dropout=0.5):
    model = Sequential()
    
    # first convolutional layer
    model.add(Convolution2D(16,5,5,border_mode='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,4,4,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.summary()    


    # setting sgd optimizer parameters
    optimizer = Adadelta() #Adagrad()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    
    return model
    
def make_binary_model(input_shape,dropout=0.25):
    model = Sequential()
    
    # first convolutional layer
    model.add(Convolution2D(16,4,4,border_mode='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,4,4,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout))

    #model.add(Convolution2D(32,4,4,border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.summary()    


    optimizer = Adadelta() #Adagrad()
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    
    return model    

def make_binary_model_full(input_shape,dropout=0.25,activation='sigmoid'):
    model = Sequential()
    
    # first convolutional layer
    #model.add(Convolution2D(16,4,4,border_mode='same',input_shape=input_shape))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(dropout))

    #model.add(Convolution2D(32,4,4,border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(dropout))

    #model.add(Convolution2D(32,4,4,border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(dropout))

    h1 = 100
    h2 = 50

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(h1,init='lecun_uniform',activation=activation))
    model.add(Dropout(dropout))

    if h2 is not None:
        model.add(Dense(h2,init='lecun_uniform',activation=activation))
        model.add(Dropout(dropout))
    
    model.add(Dense(1,activation='sigmoid',init='lecun_uniform'))

    model.summary()

    # setting sgd optimizer parameters
    optimizer = Adadelta() #Adagrad()
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    #model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    
    return model    


def train(model, X_train,X_test,y_train,y_test,nb_classes,batch_size,nb_epoch,callbacks=[],generator=None):
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')

    # pre-process training and testing data
    #mean_value = np.mean(X_train_original)
    #max_value = np.std(X_train_original)

    #X_train = (X_train_original - mean_value) / max_value
    #if X_test_original is not None:
    #    X_test = (X_test_original - mean_value) / max_value

    #print ("mean",mean_value,"max",max_value)

    #X_train = X_train.reshape(X_train.shape[0], 1, 100, 100)
    #X_test = X_test.reshape(X_test.shape[0], 1, 100, 100)

    # convert class vectors to binary class matrices
    if nb_classes > 1:
        Y_train = np_utils.to_categorical(y_train, nb_classes)
    else:
        Y_train = y_train
        
    if X_test is not None:
        if nb_classes > 1:
            Y_test = np_utils.to_categorical(y_test, nb_classes)
        else:
            Y_test = y_test
    
    if X_test is not None:
        if generator is None:
            history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=1, validation_data=(X_test, Y_test),callbacks=callbacks,shuffle=True)
        else:
            history = model.fit_generator(generator.flow(X_train, Y_train,
                            batch_size=batch_size),samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                            verbose=1, validation_data=(X_test, Y_test),callbacks=callbacks)
                        
        score = model.evaluate(X_test, Y_test, verbose=1)

        return score
    else:
        history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1,shuffle=True,callbacks=callbacks)

        return None

def test(model,X_test,y,nb_classes):
    #X_test = X.astype('float32')
    #X_test = (X - mean_value)/max_value
    #X_test = X_test.reshape(X_test.shape[0], 1, 100, 100)

    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y, nb_classes)
    return model.test_on_batch(X_test, Y_test)


def load_model(model_filename,model_weights_filename):
    model = model_from_json(model_filename)
    #model.compile()
    model.load_weights(model_weights_filename)

    return model

def save_model(model,model_filename,model_weights_filename):
    print("saving model",model_filename,model_weights_filename)
    json_string = model.to_json()
    open(model_filename, 'w').write(json_string)
    model.save_weights(model_weights_filename,overwrite=True)
