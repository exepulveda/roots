import os
import os.path

import config
import utils



def classify_new_folder(binary_model,frame_model,input_folder,output_folder,config):
    '''this function apply classifier good/bad to all images in folder and after apply frame classifier
    '''
    images = []
    utils.expand_folder(folder,images)
    
    X = utils.make_X(images,config.with,config.heigth)
    
    predictions = binary_model.predict(X)
    
    #create folder rejected/accepted images
    os.mkdir(os.path.join(output_folder,"rejected"))
    #accepted
    os.mkdir(os.path.join(output_folder,"unclassified_frame"))
    for i in range(config.start_frame,config.end_frame+1):
        os.mkdir(os.path.join(output_folder,"frame-{0}".format(i)))
        
    
    n = len(images)

    X_frames_list = [] 
    accepted_list = []

    for image,prediction in zip(images,predictions):
        if prediction == 0:
            accepted_list += [image]


            #now keep only left-up corner for frame classification
            X_frame = np.empty((n,config.frame.with * config.frame.height))
            for i in range(len(images)):
                image = X[i,:]
                image.reshape(images,config.with,config.heigth)
                X_frames_list += [image[:,:].flatten()]
                
            

            
            shutil.copy(image,dest)
        else:
            dest = os.path.join(output_folder,"accepted")
            shutil.copy(image,dest)        
            
    #frame classification
    #transform to np.array
    X_frames = np.array(X_frames_list)
    frame_predictions = frame_model.predict(X_frames)
    for image,prediction in zip(X_frames_list,predictions):
        dest = os.path.join(output_folder,"frame-{0}".format(prediction+config.start_frame)))
        os.mkdir(dest)
        shutil.copy(image,dest)        
        
    
            
if __name__ == "__main__":
    
