import os
import cv2

#PATH_VIDEO = '/home/esepulveda/Documents/projects/1.14.AVI'
#PATH_IMAGES = '/home/esepulveda/Documents/projects/tmp/images/'

def extract_frames_from_video(video_path,frames_path,skip=15,extension="tiff"):
    # Create output dir
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    vidcap = cv2.VideoCapture(video_path)

    image_list = []
    i=1
    k=1

    success = True
    while success:
        success, frame = vidcap.read()

        #print success,i,video_path
        if success:
            if (i % skip) == 0:
                image_filename = os.path.join(frames_path,'{0}.{1}'.format(k,extension))
                cv2.imwrite(image_filename, frame)
                #print "extracting",i,k,image_filename
                image_list += [image_filename]
                k += 1
            i += 1

    vidcap.release()
    return image_list,i


#if __name__ == "__main__":
#    extract_frames_from_video(PATH_VIDEO,PATH_IMAGES)
