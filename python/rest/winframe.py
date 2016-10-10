import cv2
import numpy as np

# Find a windows frame such that the error is minimised w.r.t.
# the central frame of the windows

HISTRES = 8

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


def winframe(images, win_sz=7):
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
    elif n >= 5:
        best_win = winframe(images, win_sz=5)
    else:
        best_win = images

    return best_win
