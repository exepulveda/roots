import cv2
import os.path

def image_extraction(video_filename,images_folder="."):
    video_filename_noext = path.basename(video_filename)
    print video_filename_noext
    quit()
    vidcap = cv2.VideoCapture(video_filename)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        #print 'Read a new frame: ', success
        image_filename = os.path.join(images_folder,video_filename_noext)
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        count += 1


image_extraction("/media/esepulveda/7846-95DD/Videos and stacks/1.11.AVI")
