import cv2
import numpy as np

# Find a windows frame such that the error is minimised w.r.t.
# the central frame of the windows

HISTRES = 11

def win_dist(images):
    n = len(images)
    h = cv2.calcHist(images[n / 2], [0, 1, 2], None,
                     [HISTRES, HISTRES, HISTRES],
                     [0, 256, 0, 256, 0, 256])
    cv2.normalize(h, h, 0, 255, cv2.NORM_MINMAX)

    d = 0
    for i in range(0, n):
        hi = cv2.calcHist(images[i], [0, 1, 2], None,
                          [HISTRES, HISTRES, HISTRES],
                          [0, 256, 0, 256, 0, 256])
        cv2.normalize(hi, hi, 0, 255, cv2.NORM_MINMAX)

        d += cv2.compareHist(h, hi, cv2.HISTCMP_CHISQR)

    return d


def winframe(images, win_sz=7,debug=False):
    # Find the windows frame with the lesser error

    n = len(images)

    if win_sz <= n:
        best_d = np.inf

        for i in range(0, n - win_sz + 1):
            win = images[i: i + win_sz]
            d = win_dist(win)
            if d < best_d:
                best_d = d
                best_win = win
                if debug: print i, i + win_sz
    elif n > 3:
        best_win = winframe(images,win_sz=3,debug=debug)
    else:
        best_win = images

    return best_win

# Restore images by aligning them and taking the median
# at each pixel.

def restore(images,iterations=50):

    num_of_images = len(images)
    ref_idx = num_of_images/2
 
    # Equalize all images
    for i in images:
        cv2.xphoto.balanceWhite(i, i, cv2.xphoto.WHITE_BALANCE_SIMPLE)

    # At least 3 images are needed, otherwise we just return the first one
    if num_of_images < 3:
        return images[0];

    # Register imags w.r.t. the central one
    fixed = cv2.cvtColor(images[ref_idx],cv2.COLOR_BGR2GRAY)

    # Find size
    sz = fixed.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-8;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, termination_eps)

    alignedImages=[]
    alignedImagesIntensity=[]

    i = 0
    for moving in images:

        if i == ref_idx:
            aligned=images[ref_idx]
            aligned_intensity = fixed
        else:
            try:
                # convert to intensity
                moving = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

                # Define 3x3 matrices and initialize the matrix to identity
                warp_matrix = np.eye(3, 3, dtype=np.float32)

                # Run the ECC algorithm. The results are stored in warp_matrix.
                (cc, warp_matrix) = cv2.findTransformECC(fixed, moving, warp_matrix, warp_mode, criteria)

                aligned = cv2.warpPerspective(images[i], warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            except:
                aligned = None

        if aligned is not None:
            aligned_intensity = cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)
            alignedImages.append(aligned)
            alignedImagesIntensity.append(aligned_intensity)

        i += 1

    nx = sz[0]
    ny = sz[1]
    nz = num_of_images

    retr = np.empty((nx,ny))
    retg = np.empty((nx,ny))
    retb = np.empty((nx,ny))
    for i in xrange(nx):
        for j in xrange(ny):
            ii = None
            values = [x[i,j] for x in alignedImagesIntensity]
            values.sort()
            center = values[nz//2]
            for k in xrange(nz):
                if alignedImagesIntensity[k][i,j] == center:
                   ii = k
                   break

            assert ii is not None
            retb[i,j] = alignedImages[ii][i,j,0] 
            retg[i,j] = alignedImages[ii][i,j,1] 
            retr[i,j] = alignedImages[ii][i,j,2] 
  
    retb = np.uint8(retb)
    retg = np.uint8(retg)
    retr = np.uint8(retr)

    restored = cv2.merge((retb,retg,retr))
    return restored


def find_the_most_similar(images_data):
    from skimage.measure import compare_ssim

    n = len(images_data)

    sim = np.zeros((n,n))

    for i in xrange(n):
        image_i = images_data[i]
        for j in xrange(i,n):
            image_j = images_data[j]
            sim[i,j] = compare_ssim(image_i,image_j,multichannel=True) #,gaussian_weights=True)
            sim[j,i] = sim[i,j]

    ret = np.sum(sim,axis=0)
    
    ret = zip(ret,range(n))
    ret.sort()
    
    return ret[-1][1]


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

    
