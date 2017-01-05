import glob, os
from restore import *
from winframe import *

# Set input and output paths
<<<<<<< HEAD
<<<<<<< HEAD
DATA_PATH = '/home/esepulveda/Documents/projects/roots/python/processing/1.25.AVI/windows/frame-50/' 
=======
DATA_PATH = '/Users/a1613915/Dropbox/alvaro-projects/selection-wrong/1.25/windows/frame-29/' 
>>>>>>> 404ce3cbe4fce89a15da17c55c0227f718f0b06e
=======
DATA_PATH = '/Users/a1613915/Dropbox/alvaro-projects/selection-wrong/1.25/windows/frame-50/' 

>>>>>>> 62cc7d02358419c8b92af3487edb7ce9b46addec
OUT_PATH = 'results'

# Create output dir
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# Retrieve frames
#frames = glob.glob( '{}/*'.format(DATA_PATH) )


for framename in [DATA_PATH]:
    # Load the images of the frame
    inames = glob.glob('{}/*.tiff'.format(framename))
    
    inames.sort()
    
    #inames.sort()
    #print inames
    numOfImages=len(inames)
    images = [cv2.imread(inames[i]) for i in range(numOfImages)]

<<<<<<< HEAD
=======
    wf = winframe(images)
>>>>>>> 62cc7d02358419c8b92af3487edb7ce9b46addec

    # Find the best windows frame
    wf = winframe(images,debug=True)
    print len(wf)
    
    if len(wf) >= 5:

        # Restore images from the windows frame
        restored = restore(wf)
    else:
        ret = find_the_most_similar(images)
        restored = images[ret]

    # Balance whites
    #cv2.xphoto.balanceWhite(restored, restored, cv2.xphoto.WHITE_BALANCE_SIMPLE)

    # Show result
    cv2.imshow("Restored", restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    name = framename.split('/')[-1]
    cv2.imwrite('{}/restored_{}.tiff'.format(OUT_PATH,name), restored)
