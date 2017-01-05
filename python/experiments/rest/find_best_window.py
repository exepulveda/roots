import cv2
from restore import restore, find_the_most_similar
from winframe import winframe

def select_and_restore(image_names,destination,configuration):
    image_names.sort()
    
    numOfImages=len(image_names)
    
    
    images = [cv2.imread(image_names[i]) for i in range(numOfImages)]


    # Find the best windows frame
    wf = winframe(images)

    if len(wf) >= 5:
        # Restore images from the windows frame
        restored = restore(wf,configuration.restore.iterations)
    else:
        ret = find_the_most_similar(images)
        restored = images[ret]

    # Restore images from the windows frame

    # balance whites
    #cv2.xphoto.balanceWhite(restored, restored, cv2.xphoto.WHITE_BALANCE_SIMPLE)

    # Show result
    # cv2.imshow("Restored", restored)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save result
    cv2.imwrite(destination, restored)

    
