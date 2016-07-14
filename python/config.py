import os.path
import shutil

PROJECT_HOME = "."

class Dummy():
    pass

configuration = Dummy()

configuration.home = PROJECT_HOME
configuration.tmp_dir = "/tmp"
configuration.max_images = 100
configuration.frame_step = 1 #how many frames are skipped from video
configuration.extension = "tiff"


configuration.model = Dummy()
configuration.model.classifiers = {
    'final': os.path.join(configuration.home,"models/model-binary"),
    'cv-0' : os.path.join(configuration.home,"models/model-binary-cv-0"),
    'window-final': os.path.join(configuration.home,"models/model-window"),
    }
configuration.model.classifier = configuration.model.classifiers['cv-0']

configuration.model.classifier_weights = os.path.join(configuration.home,"models/model-binary.h5")
configuration.model.classifier_mean = 127.367759008 #mean_value 
configuration.model.classifier_max = 43.5644119735
configuration.model.window = configuration.model.classifiers['window-final']
#configuration.model.window_mean = 166.169432427  #166.166054305 max 46.4016257135
#configuration.model.window_max = 46.3409904943 #mean 166.169432427 max 46.3409904943 166.169432427 max 46.3409904943

configuration.input = Dummy()
configuration.input.image_with = 384
configuration.input.image_height = 288

configuration.window = Dummy()
configuration.window.image_with = 96 #96
configuration.window.image_height = 72 #72
configuration.window.offset_with = 10 #10
configuration.window.offset_height = 10 #10
configuration.window.offset_class = 6 #classifier start in 0
configuration.window.start = 6
configuration.window.end = 54

configuration.restore = Dummy()
configuration.restore.iterations = 30

configuration.roofly = Dummy()
configuration.roofly.template = "EXPERIMENT_T{tube}_L{window}_{date}_{time}_{session}_CBD.{extension}"

def get_configuration():
    return configuration
    
def setup_video_folders(video_filname):
    #extract video name
    drive, path = os.path.splitdrive(video_filname)
    path, video_name = os.path.split(path)
    
    #create video folders
    if not os.path.exists(os.path.join(configuration.home,"processing")):
        os.mkdir(os.path.join(configuration.home,"processing"))
        
    video_folder = os.path.join(configuration.home,"processing",video_name)

    #delete video folder
    shutil.rmtree(video_folder,ignore_errors=True)


    if not os.path.exists(video_folder):
        os.mkdir(video_folder)

    if not os.path.exists(os.path.join(video_folder,"allframes")):
        os.mkdir(os.path.join(video_folder,"allframes"))
    if not os.path.exists(os.path.join(video_folder,"rejected")):
        os.mkdir(os.path.join(video_folder,"rejected"))
    if not os.path.exists(os.path.join(video_folder,"accepted")):
        os.mkdir(os.path.join(video_folder,"accepted"))
    if not os.path.exists(os.path.join(video_folder,"selected")):
        os.mkdir(os.path.join(video_folder,"selected"))
    if not os.path.exists(os.path.join(video_folder,"windows")):
        os.mkdir(os.path.join(video_folder,"windows"))

        
    return video_folder

def get_video_folder(video_name):
    video_folder = os.path.join(configuration.home,"processing",video_name)
    return video_folder
