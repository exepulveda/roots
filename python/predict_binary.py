from __future__ import print_function
import utils
import os
import os.path
import random
import json

import mlp

def set_cross_validation(folders):
    '''Define a list with all videos. Folders contains a list of paths
    to each video. Each folder has a folder bads and oks. Inside oks folder with frame images
    '''
    images = []
    for images_path in folders:
        #bads images
        filenames = []
        utils.expand_folder(os.path.join(images_path,"bads"),filenames)
        bads_filenames = [(x,0) for x in filenames]
        #oksimages
        oks_filenames = []
        for i in range(6,55):
            #making folder name
            filenames = []
            utils.expand_folder(os.path.join(images_path,"oks","frame-{0}".format(i)),filenames)
            oks_filenames += [(x,i) for x in filenames]
            
        
        images += [bads_filenames + oks_filenames] 
        
    return images

if __name__ == "__main__":
    model = mlp.load_model("models/model-convnet-0.json","models/model-convnet-0.h5")        

    images_path = "../training/1.11"
    oks_filenames = []
    for i in range(6,55):
        #making folder name
        filenames = []
        utils.expand_folder(os.path.join(images_path,"oks","frame-{0}".format(i)),filenames)
        oks_filenames += [(x,i-6) for x in filenames]

    
    w = 384
    h = 288
    
    offset_w = 10
    offset_h = 10
        
    target_w = w // 4
    target_h = h //4
    
    mean_image = 0.652293610732
    max_image = 0.181551718176
    
    for i,(image,tag) in enumerate(oks_filenames):
        #X_train = utils.load_image(image,384,288)
        
        X = utils.load_image_convnet(image,w,h,offset=[offset_w,offset_h],size=[target_w,target_h])
        
        X -= mean_image
        X /= max_image
        
        ret = model.predict_classes(X, batch_size=1, verbose=0)
        
        print(image,ret,tag)
        
