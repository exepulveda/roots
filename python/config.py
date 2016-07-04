import os.path

PROJECT_HOME = "."

class Dummy():
    pass

configuration = Dummy()

configuration.home = PROJECT_HOME
configuration.start_frame = 6
configuration.end_frame = 54

configuration.model = Dummy()
configuration.model.classifier = os.path.join(configuration.home,"models/binary.json") #"models/model-convnet-1.json","models/model-convnet-1.json"
configuration.model.classifier_weights = os.path.join(configuration.home,"models/binary.h5")
configuration.model.classifier_mean = 0.11
configuration.model.classifier_max = 4.5
configuration.model.frames = os.path.join(configuration.home,"models/model-convnet-0.json")
configuration.model.frames_weights = os.path.join(configuration.home,"models/model-convnet-0.h5")
configuration.model.frames_mean = 166.169432427  #166.166054305 max 46.4016257135
configuration.model.frames_max = 46.3409904943 #mean 166.169432427 max 46.3409904943

configuration.input = Dummy()
configuration.input.image_with = 384
configuration.input.image_height = 288

configuration.frame = Dummy()
configuration.frame.image_with = 96
configuration.frame.image_height = 72
configuration.frame.offset_with = 10
configuration.frame.offset_height = 10
configuration.frame.offset_class = 6 #classifier start in 0

configuration.roofly = Dummy()
configuration.roofly.template = "EXPERIMENT_T{tube}_L{window}_{date}_{time}_{session}_CBD.jpg"

def get_configuration():
    return configuration
    
