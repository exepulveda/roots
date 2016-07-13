import os
import os.path
import sklearn.cross_validation
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

bad_path = "/home/esepulveda/Dropbox/Roots videos/Videos and stacks/Deleted stacks"
good_path = "/home/esepulveda/Dropbox/Roots videos/Videos and stacks/Stacks"


def expand_folder(path,container):
    for dirpath, dirnames, files in os.walk(path):
        #print dirpath,len(dirnames),len(files)
        #for name in files:
        #    filename = os.path.join(dirpath, name)
        #    print filename
        #    files += [filename]
        for name in files:
            container += [os.path.join(dirpath,name)]
            
good_files = []
expand_folder(good_path,good_files)
print len(good_files)

bad_files = []
expand_folder(bad_path,bad_files)
print len(bad_files)

#add label
good_files_labeled = [(x,1) for x in good_files]
bad_files_labeled = [(x,0) for x in bad_files]

#all together
files_labeled = good_files_labeled + bad_files_labeled

training_files, testing_files = sklearn.cross_validation.train_test_split(files_labeled,train_size=0.8,random_state=1634120)

image_shape = (288,384,3)

X = np.empty((len(training_files),288,384))
y = np.empty(len(training_files),dtype=np.int32)

#built training X and y
for i,(image_filename,tag) in enumerate(training_files):
    #print image_filename,tag
    image = Image.open(image_filename).convert('L')
    image = np.array(image) / 255.0

    X[i,:,:] = image
    y[i] = tag


np.save('training.X',X)
np.save('training.y',y)

X = np.empty((len(testing_files),288,384))
y = np.empty(len(testing_files),dtype=np.int32)

#built training X and y
for i,(image_filename,tag) in enumerate(testing_files):
    #print image_filename,tag
    image = Image.open(image_filename).convert('L')
    image = np.array(image) / 255.0

    X[i,:,:] = image
    y[i] = tag


np.save('testing.X',X)
np.save('testing.y',y)
