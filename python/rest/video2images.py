import os
import cv2

PATH_VIDEO = '/home/esepulveda/Documents/projects/1.14.AVI'
PATH_IMAGES = '/home/esepulveda/Documents/projects/tmp/images/'

def extract_frames_from_video(video_path,frames_path,skip=1):
    # Create output dir
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)


    cap = cv2.VideoCapture(video_path)

<<<<<<< HEAD
i = 1
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('{}/{}.tiff'.format(PATH_IMAGES,i), frame)
    i += 1
=======
    image_list = []
    i=1
    k=1
    while cap.isOpened():
        ret, frame = cap.read()
        if (i % skip) == 0:
            image_filename = '{}/{}.jpg'.format(PATH_IMAGES,k)
            cv2.imwrite(image_filename, frame)
            image_list += [image_filename]
            k += 1
        i += 1
>>>>>>> 4f60fc13355b66af4f741a6bd0060f21f6000222

    cap.release()
    return image_list


if __name__ == "__main__":
    extract_frames_from_video(PATH_VIDEO,PATH_IMAGES)
