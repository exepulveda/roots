import numpy as np
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t


#class LineLocation:
#    N, S, W, E = range(4)


#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
def quality_cy(np.ndarray[DTYPE_t, ndim=2] im, float cx, float cy, float r, int location):
    cdef int h = im.shape[0]
    cdef int w = im.shape[1]

    cdef float xmin
    cdef float xmax
    cdef float ymin
    cdef float ymax


    cdef int count1
    cdef int count2
    cdef float val1
    cdef float val2

    cdef int xp
    cdef int yp

    cdef float r2 = r*r
    

    
    cdef max_size = max(w, h)

    cdef int r_min
    cdef int r_max


    #print h,w,cx,cy,r,location


    if location >= 2:

        ymin = cy - r
        ymax = cx + r

        val1 = 0.0
        val2 = 0.0
        count1 = 0
        count2 = 0

        r_min = max(0, int(ymin))
        r_max = min(h, int(ymax))

        for yp in range(r_min, r_max):
            xp = int(cx + sqrt(r2 - (yp - cy)**2))
            
            if (xp >= 0) and  (xp < w):
                val1 = val1 + im[yp,xp]
                count1 = count1 + 1

            xp = int(cx - sqrt(r2 - (yp - cy)**2))
            
            if (xp >= 0) and  (xp < w):
                val2 = val2 + im[yp,xp]
                count2 = count2 + 1

        val1 = val1 / max(max_size, count1)
        val2 = val2 / max(max_size, count2)

        return max(val1,val2)

    else:

        xmin = cx - r
        xmax = cx + r

        val1 = 0.0
        val2 = 0.0
        count1 = 0
        count2 = 0

        r_min = max(0, int(xmin))
        r_max = min(w, int(xmax))

        for xp in range(r_min, r_max):
            yp = int(cy + sqrt(r2 - (xp - cx)**2))
            
            if (yp >= 0) and  (yp < h):
                val1 = val1 + im[yp,xp]
                count1 = count1 + 1

            yp = int(cy - sqrt(r2 - (xp - cx)**2))
            
            if (yp >= 0) and  (yp < h):
                val2 = val2 + im[yp,xp]
                count2 = count2 + 1

        val1 = val1 / max(max_size, count1)
        val2 = val2 / max(max_size, count2)

        return max(val1,val2)
