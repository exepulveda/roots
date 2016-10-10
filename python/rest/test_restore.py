import glob, os
from restore import *
from winframe import *

# Set input and output paths
DATA_PATH = '/Users/a1613915/Dropbox/alvaro-projects/selection-wrong/1.25/windows/frame-51/' 
OUT_PATH = 'results'

# Create output dir
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# Retrieve frames
#frames = glob.glob( '{}/*'.format(DATA_PATH) )


for framename in [DATA_PATH]:
    # Load the images of the frame
    inames = glob.glob('{}/*.tiff'.format(framename))
    numOfImages=len(inames)
    images = [cv2.imread(inames[i]) for i in range(numOfImages)]

    # Find the best windows frame
    wf = winframe(images)

    # Restore images from the windows frame
    restored = restore(wf)

    # Balance whites
    #cv2.xphoto.balanceWhite(restored, restored, cv2.xphoto.WHITE_BALANCE_SIMPLE)

    # Show result
    cv2.imshow("Restored", restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    name = framename.split('/')[-1]
    cv2.imwrite('{}/restored_{}.tiff'.format(OUT_PATH,name), restored)
