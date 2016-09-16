import cv2
from restore import restore
from winframe import winframe

def select_and_restore(image_names,destination,configuration):
    numOfImages=len(image_names)
    images = [cv2.imread(image_names[i]) for i in range(numOfImages)]

    # Find the best windows frame
    wf = winframe(images)

    # Restore images from the windows frame
    restored = restore(wf,configuration.restore.iterations)

    # balance whites
    #cv2.xphoto.balanceWhite(restored, restored, cv2.xphoto.WHITE_BALANCE_SIMPLE)

    # Show result
    # cv2.imshow("Restored", restored)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save result
    cv2.imwrite(destination, restored)
