import glob, os
from restore import *
from winframe import *

DATA_PATH = '../../1.11/oks'
OUT_PATH = 'results'

# Create output dir
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# Retrieve frames
frames = glob.glob( '{}/frame-*'.format(DATA_PATH) )

for framename in frames:

    # Load the images of the frame
    inames = glob.glob('{}/*.jpg'.format(framename))
    numOfImages=len(inames)
    images = [cv2.imread(inames[i]) for i in range(numOfImages)]

    # Find the best windows frame
    wf = winframe(images)

    # Restore images from the windows frame
    restored = restore(wf)

    # balance whites
    #cv2.xphoto.balanceWhite(restored, restored, cv2.xphoto.WHITE_BALANCE_SIMPLE)

    # Show result
    # cv2.imshow("Restored", restored)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save result
    name = framename.split('/')[-1]
    cv2.imwrite('{}/restored_{}.tiff'.format(OUT_PATH,name), restored)