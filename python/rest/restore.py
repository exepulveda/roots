import cv2
import numpy as np

# Restore images by aligning them and taking the median
# at each pixel.

def restore(images,iterations=50):

    num_of_images = len(images)
    ref_idx = num_of_images/2

    # Register images
    fixed = cv2.cvtColor(images[ref_idx],cv2.COLOR_BGR2GRAY)

    # Find size
    sz = fixed.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Specify the number of iterations.
    numOfIterations = iterations;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-8;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, numOfIterations, termination_eps)

    alignedImages=[]

    i = 0
    for moving in images:
        if i == ref_idx:
            aligned=images[ref_idx]
        else:
            # convert to intensity
            moving = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

            # Define 3x3 matrices and initialize the matrix to identity
            warp_matrix = np.eye(3, 3, dtype=np.float32)

            #print('processing...\n')
            # Run the ECC algorithm. The results are stored in warp_matrix.
            (cc, warp_matrix) = cv2.findTransformECC(fixed, moving, warp_matrix, warp_mode, criteria)

            aligned = cv2.warpPerspective(images[i], warp_matrix, (sz[1], sz[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        alignedImages.append(aligned)

        i += 1

    # split images in r,g,b channels
    b = np.array([img[:,:,0] for img in alignedImages])
    g = np.array([img[:,:,1] for img in alignedImages])
    r = np.array([img[:,:,2] for img in alignedImages])

    # restore image using the median pixel
    b = np.uint8(np.median(b,0));
    g = np.uint8(np.median(g,0));
    r = np.uint8(np.median(r,0));

    restored = cv2.merge((b,g,r))
    return restored

