import os
import os.path
import sklearn.cross_validation
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

working_path = "/home/esepulveda/Documents/projects/roots"

bad_path = os.path.join(working_path,"1.11/bads")
good_path = os.path.join(working_path,"1.11/oks")


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

bad_files = []
expand_folder(bad_path,bad_files)

ret = []
for filename in good_files:
    ret += [(filename, 1)]

for filename in bad_files:
    ret += [(filename, 2)]
    
training, testing = sklearn.cross_validation.train_test_split(ret,train_size=0.8,random_state=1634120)

print "Training"
for filename,tag in training:
    print filename,tag


print "Testing"
for filename,tag in testing:
    print filename,tag

