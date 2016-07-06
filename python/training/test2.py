import os
import os.path
import sklearn.cross_validation
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

print len(files_labeled)

training_files, testing_files = sklearn.cross_validation.train_test_split(files_labeled,train_size=0.8,random_state=1634120)

image_shape = (288,384,3)

X = np.empty((len(training_files),140*140))
y = np.empty(len(training_files),dtype=np.int32)

pixel_crop = 70

#built training X and y
for i,(image_filename,tag) in enumerate(training_files):
    #print image_filename,tag
    image = mpimg.imread(image_filename) / 255.0
    
    #select only four corners
    new_img = np.zeros((pixel_crop*2,pixel_crop*2,3))
    
    new_img[0:pixel_crop,0:pixel_crop,:] = image[0:pixel_crop,0:pixel_crop,:]
    new_img[0:pixel_crop,-pixel_crop:,:] = image[0:pixel_crop,-pixel_crop:,:]
    new_img[-pixel_crop:,0:pixel_crop,:] = image[-pixel_crop:,0:pixel_crop,:]
    new_img[-pixel_crop:,-pixel_crop:,:] = image[-pixel_crop:,-pixel_crop:,:]
    
    #print image.shape
    #plt.imshow(new_img)
    #plt.imshow(image)
    #plt.show()
    #quit()
    
    X[i,:] = new_img[:,:,0].flatten()
    y[i] = tag


from mlxtend.tf_classifier import TfMultiLayerPerceptron

mlp = TfMultiLayerPerceptron(eta=0.005, 
                             epochs=100, 
                             hidden_layers=[100], #,20],
                             activations=['logistic'], #,'logistic'],
                             print_progress=3, 
                             minibatches=1, 
                             optimizer='adam',
                             random_seed=1)

mlp.fit(X, y)

plt.plot(range(len(mlp.cost_)), mlp.cost_)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

#test
ntest = len(testing_files)
X = np.empty((ntest,pixel_crop*2*pixel_crop*2))
y = np.empty(ntest,dtype=np.int32)
for i,(image_filename,tag) in enumerate(testing_files):
    #print image_filename,tag
    image = mpimg.imread(image_filename) / 255.0
    
    new_img = np.empty((pixel_crop*2,pixel_crop*2,3))
    
    new_img[0:pixel_crop,0:pixel_crop,:] = image[0:pixel_crop,0:pixel_crop,:]
    new_img[0:pixel_crop,-pixel_crop:,:] = image[0:pixel_crop,-pixel_crop:,:]
    new_img[-pixel_crop:,0:pixel_crop,:] = image[-pixel_crop:,0:pixel_crop,:]
    new_img[-pixel_crop:,-pixel_crop:,:] = image[-pixel_crop:,-pixel_crop:,:]
    
    X[i,:] = new_img[:,:,0].flatten()
    y[i] = tag
    
ret = mlp.predict(X)

print "\nprediction:",ntest

print np.sum(y==ret)
print np.sum(y==ret)/float(ntest) * 100.0

#print zip(y,ret)
