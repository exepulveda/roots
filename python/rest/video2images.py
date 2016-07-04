import os
import cv2

PATH_VIDEO = '/Users/a1613915/Dropbox/Roots videos/1.14.AVI'
PATH_IMAGES = './images/'

# Create output dir
if not os.path.exists(PATH_IMAGES):
    os.makedirs(PATH_IMAGES)


cap = cv2.VideoCapture(PATH_VIDEO)

i=1
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('{}/{}.tiff'.format(PATH_IMAGES,i), frame)
    i += 1

cap.release()
