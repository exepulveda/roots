import os.path
import os
import shutil
import json

class Dummy():
    pass

configuration = Dummy()

configuration.home = os.environ.get('PROJECT_HOME','.')
configuration.tmp_dir = "/tmp"
configuration.max_images = 100
configuration.frame_step = 1 #how many frames are skipped from video
configuration.extension = "tiff"
configuration.batch_size = 1000

configuration.model = Dummy()
configuration.model.home = os.environ.get('MODEL_HOME','models')

configuration.model.classifier = os.path.join( configuration.model.home, 'model_binary')
configuration.model.window     = os.path.join( configuration.model.home, 'model_window')
configuration.model.window_templates = os.path.join(configuration.model.home,"templates")
configuration.model.min_probability = 0.90

configuration.input = Dummy()
configuration.input.image_width = 384
configuration.input.image_height = 288
configuration.input.binary_width = 384 / 4
configuration.input.binary_height = 288 /4

configuration.window = Dummy()
configuration.window.image_width = 96 #96
configuration.window.image_height = 72 #72
configuration.window.offset_width = 10 #10   
configuration.window.offset_height = 10 #10
configuration.window.offset_class = 6 #classifier start in 0
configuration.window.start = 6
configuration.window.end = 54

configuration.restore = Dummy()
configuration.restore.iterations = 30

configuration.rectify = Dummy()
configuration.rectify.image_width = 400
configuration.rectify.image_height = 300
configuration.rectify.iterations = 7500

configuration.rootfly = Dummy()
configuration.rootfly.template = "T{tube}_L{window:03d}_{year:04d}.{month:02d}.{day:02d}_{session:03d}.jpg" 
configuration.rootfly.to_copy_from = os.path.join(configuration.home,"processing","{tube}-{date}.AVI","selected","frame-*." + configuration.extension)

def get_configuration():
    return configuration
    
    
def get_processed_foldername():
    if not os.path.exists(os.path.join(configuration.home,"processed")):
        os.mkdir(os.path.join(configuration.home,"processed"))

    return os.path.join(configuration.home,"processed")

def get_processing_foldername():
    if not os.path.exists(os.path.join(configuration.home,"processing")):
        os.mkdir(os.path.join(configuration.home,"processing"))

    return os.path.join(configuration.home,"processing")

    
def setup_video_folders(video_filname,reset_folder=True):
    #extract video name
    drive, path = os.path.splitdrive(video_filname)
    path, video_name = os.path.split(path)
    
    #create video folders
    if not os.path.exists(os.path.join(configuration.home,"processing")):
        os.mkdir(os.path.join(configuration.home,"processing"))
        
    video_folder = os.path.join(configuration.home,"processing",video_name)

    #delete video folder
    if reset_folder:
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

def save_video_status(video_filename,status):
    status_filename = os.path.join(video_filename,"status.txt")

    with open(status_filename,"w") as fout:
        json.dump(status,fout)

def load_video_status(video_filename):
    try:
        status_filename = os.path.join(video_filename,"status.txt")
        
        if os.path.exists(status_filename):
            with open(status_filename,"r") as fin:
                obj = json.load(fin)
                return obj
        else:
            return json.loads("{}")
    except:
        return json.loads("{}")

def get_video_folder(video_name):
    video_folder = os.path.join(configuration.home,"processing",video_name)
    return video_folder
