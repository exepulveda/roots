# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:56:23 2016

@author: aparra
"""

import numpy as np
import cv2
from rbfwarp2d import rbfwarp2d

#from quality_cython import quality_cy

norm = np.linalg.norm
epsilon = 0.000000001

class Circle:
    """Circle representation"""

    def __init__(self, centre=np.array([0, 0]), rad=0):
        self.centre = centre
        self.rad = rad

    @property
    def centre(self):
        return self.centre

    @property
    def radius(self):
        return self.rad

    @staticmethod
    def from_base(base):
        # type: (ndarray) -> Circle

        centre = np.array([0.0, 0.0])

        pt1 = base[:, 0]
        pt2 = base[:, 1]
        pt3 = base[:, 2]

        delta_a = pt2 - pt1
        delta_b = pt3 - pt2

        ax_is_0 = abs(delta_a[0]) <= epsilon
        bx_is_0 = abs(delta_b[0]) <= epsilon

        # check whether both lines are vertical - collinear
        if ax_is_0 and bx_is_0:
            print("WARNING: Points are on a straight line (collinear).")
            return Circle(centre, -1)

        # make sure delta gradients are not vertical
        # swap points to change deltas
        if ax_is_0:
            pt2, pt3 = pt3, pt2
            delta_a = pt2 - pt1

        if bx_is_0:
            tmp = pt1
            pt1 = pt2
            pt2 = tmp
            delta_b = pt3 - pt2

        grad_a = delta_a[1] / delta_a[0]
        grad_b = delta_b[1] / delta_b[0]

        # check whether the given points are collinear
        if abs(grad_a - grad_b) <= epsilon:
            print("WARNING: Points are on a straight line (collinear).")
            return Circle(centre, -1)

        # swap grads and points if grad_a is 0
        if abs(grad_a) <= epsilon:
            tmp = grad_a
            grad_a = grad_b
            grad_b = tmp
            tmp = pt1
            pt1 = pt3
            pt3 = tmp

        # calculate centre - where the lines perpendicular to the centre of
        # segments a and b intersect.

        centre[0] = (grad_a * grad_b * (pt1[1] - pt3[1]) + grad_b * (pt1[0] + pt2[0]) - grad_a * (pt2[0] + pt3[0])) / (
        2. * (grad_b - grad_a))
        centre[1] = ((pt1[0] + pt2[0]) / 2. - centre[0]) / grad_a + (pt1[1] + pt2[1]) / 2.

        radius = norm(centre - pt1)
        return Circle(centre, radius)


def circle_vectorized(points):

    n = points.shape[2]

    points_prima = points - points[0, :, :]
    points_prima_squared = np.sum(points_prima ** 2, axis=1)
    # print "points_prima.shape",points_prima.shape
    # print "points_prima_squared.shape",points_prima_squared.shape

    Dp = 2 * (points_prima[1, 0] * points_prima[2, 1] - points_prima[1, 1] * points_prima[2, 0])
    # print "Dp.shape",Dp.shape

    # print "points_prima_squared",points_prima_squared,"Dp",Dp

    Up = np.empty((2, n))

    Up[0] = (points_prima[2, 1] * points_prima_squared[1] - points_prima[1, 1] * points_prima_squared[2]) / Dp
    Up[1] = (points_prima[1, 0] * points_prima_squared[2] - points_prima[2, 0] * points_prima_squared[1]) / Dp

    # print Up[:,:5]

    centers = Up + points[0, :, :]

    # distance of centers to first point
    diff = points[0, :, :] - centers
    radii = norm(diff, axis=0)

    return centers, radii




class LineLocation:
    N, S, W, E = range(4)


def improfile_vectorized(im, x, y):
    h, w = im.shape

    n,m = x.shape
    
    # cast to int
    x = np.int32(x)
    y = np.int32(y)
    
    rets = np.empty(m)
    
    for i in xrange(m):
        # remove points out of the image
        idx = (x[:,i] >= 0) * (x[:,i] < w) * (y[:,i] >= 0) * (y[:,i] < h)
        x_ = x[idx,i]
        y_ = y[idx,i]

        val = im[y_,x_]
        
        ret = np.sum(val) / (np.max([w, h, val.shape[0]]))
        
        rets[i] = ret
    return rets

def improfile_loop(im, x, y):

    # cast to int
    x = x.round().astype(np.int)
    y = y.round().astype(np.int)

    # remove points out of the image
    h, w = im.shape
    idx = (x >= 0) * (x < w) * (y >= 0) * (y < h)
    x_ = x[idx]
    y_ = y[idx]

    val = np.float32(im[y_,x_])

    val = np.sum(val) / (np.max([w, h, val.shape[0]]))
    return val


def improfile_a(im, x, y):
    # remove nan values and cast to int
    idx = ~np.isnan(x) & ~np.isnan(y)
    x = x[idx].round().astype(np.int)
    y = y[idx].round().astype(np.int)

    # remove points out of the image
    h, w = im.shape
    idx = (x >= 0) * (x < w) * (y >= 0) * (y < h)
    x = x[idx]
    y = y[idx]

    coord = zip(y, x)
    val = np.float32([im[i] for i in coord])

    val_bin = val>150
    segment_len = 0
    best_segment_len = 0
    for i in range(len(val_bin)):
        if val_bin[i]:
            segment_len += 1
        else:
            best_segment_len = max(best_segment_len, segment_len)
            segment_len = 0

    return best_segment_len


def improfile_b(im, x, y):
    # remove nan values and cast to int
    # idx = ~np.isnan(x) & ~np.isnan(y)
    #x = x[idx].round().astype(np.int)
    #y = y[idx].round().astype(np.int)

    x = x.round().astype(np.int)
    y = y.round().astype(np.int)

    # remove points out of the image
    h, w = im.shape

    val = 0.0

    for i,ix in enumerate(x):
        if ix >= 0 and ix < w:
            iy_min = max(0,y[i] - 1)
            iy_max = min(h,y[i] + 1)
            #print 'iy_min, iy_max', iy_min, iy_max, ix
            val += np.sum(im[iy_min:iy_max,ix])

    return val

def quality_vectorized(im, centres, radii,location):
    h, w = im.shape

    #centre = circle.centre
    #r = circle.radius
    
    p,n = centres.shape
    
    r2 = radii**2

    if location == LineLocation.W or location == LineLocation.E:
        #global min and max
        ymin = int(np.min(np.clip(centres[1,:] - radii,0,h)))
        ymax = int(np.max(np.clip(centres[1,:] - radii,0,h)))
        
        yp = np.zeros((n,ymax-ymin)) + np.arange(ymin, ymax)
        yp = yp.T

        xp = np.sqrt(r2 - (yp - centres[1,:])**2) + centres[0,:]
        val1 = improfile_vectorized(im, xp, yp)
        
        xp =  -np.sqrt(r2 - (yp - centres[1,:])**2) + centres[0,:]
        val2 = improfile_vectorized(im, xp, yp)

        return np.maximum(val1,val2)
        
    else:
        #global min and max
        xmin = int(np.min(np.clip(centres[0,:] - radii,0,w)))
        xmax = int(np.max(np.clip(centres[0,:] - radii,0,w)))

        xp = np.zeros((n,xmax-xmin)) + np.arange(xmin, xmax)
        xp = xp.T

        yp = np.sqrt(r2 - (xp - centres[0,:]) ** 2) + centres[1,:]
        val1 = improfile_vectorized(im, xp, yp)

        yp = -np.sqrt(r2 - (xp - centres[0,:]) ** 2) + centres[1,:]
        val2 = improfile_vectorized(im, xp, yp)

        return np.maximum(val1,val2)

def quality(im, centre,r,r2, location):
    h, w = im.shape

    if location == LineLocation.W or location == LineLocation.E:
        ymin = centre[1] - r
        ymax = centre[1] + r

        yp = np.arange(max(0, ymin), min(h, ymax))
        xp = centre[0] + np.sqrt(r2 - (yp - centre[1])**2)

        val1 = improfile_loop(im, xp, yp)

        xp = centre[0] - np.sqrt(r2 - (yp - centre[1])**2)
        val2 = improfile_loop(im, xp, yp)

    else:
        xmin = centre[0] - r
        xmax = centre[0] + r

        xp = np.arange(max(0, xmin), min(w, xmax))
        yp = centre[1] + np.sqrt(r2 - (xp - centre[0]) ** 2)
        val1 = improfile_loop(im, xp, yp)

        yp = centre[1] - np.sqrt(r2 - (xp - centre[0])**2)
        val2 = improfile_loop(im, xp, yp)

    return max(val1, val2)

def quality_inefficient(im, circle, location):
    h, w = im.shape

    centre = circle.centre
    r = circle.radius

    if location == LineLocation.W or location == LineLocation.E:
        ymin = centre[1] - r
        ymax = centre[1] + r

        yp = np.arange(max(0, ymin), min(h, ymax))
        xp = centre[0] + np.sqrt(r**2 - (yp - centre[1])**2)

        val1 = improfile_loop(im, xp, yp)

        xp = centre[0] - np.sqrt(r**2 - (yp - centre[1])**2)
        val2 = improfile_loop(im, xp, yp)

    else:
        xmin = centre[0] - r
        xmax = centre[0] + r

        xp = np.arange(max(0, xmin), min(w, xmax))
        yp = centre[1] + np.sqrt(r**2 - (xp - centre[0]) ** 2)
        val1 = improfile_loop(im, xp, yp)

        yp = centre[1] - np.sqrt(r**2 - (xp - centre[0])**2)
        val2 = improfile_loop(im, xp, yp)

    return max(val1, val2)


# def auto_canny(image, sigma=0.33):
#     # method in http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
#
#     # compute the median of the single channel pixel intensities
#     v = np.median(image)
#
#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper)
#
#     return edged

def auto_canny(im):
    # http://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    otsu_th = cv2.threshold(im,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    edged = cv2.Canny(im, otsu_th/3., otsu_th)
    return edged


def fit_circle_vectorized(im, location, iterations, debug=False):

    h, w = im.shape

    if debug: print "image shape",im.shape,"location",location

    vertical_sampling = location == LineLocation.W or location == LineLocation.E


    #create sample of circles
    ncircles = iterations
    
    efective_circles = 0 
    
    centers = np.empty((2,0))
    radii = np.empty(0)
    
    while efective_circles < iterations:
        if vertical_sampling:  # W E
            base = w * np.random.uniform(size=(3,2,ncircles))
            min_radius = 1.5 * h
            y = np.array([h / 6., h / 2., 5 * h / 6.])
            mu = (w / 2.) * np.ones(3)
            sigma = .5 * w

            base[0,1,:] = y[0]
            base[1,1,:] = y[1]
            base[2,1,:] = y[2]
            
            #x = w * np.random.rand(3)
            #base[:,0,:] = w * np.random.uniform(size=(3,ncircles))

        else:  # N S
            base = h * np.random.uniform(size=(3,2,ncircles))
            min_radius = 1.5 * w
            x = np.array([w / 6., w / 2., 5 * w / 6.])

            base[0,0,:] = x[0]
            base[1,0,:] = x[1]
            base[2,0,:] = x[2]



            mu = (h / 2.) * np.ones(3)
            sigma = .5 * h
            #y = h * np.random.rand(3)
            #base[:,1,:] = h * np.random.uniform(size=(3,ncircles))


        centers_, radii_ = circle_vectorized(base)
        
        #check for minimum radii
        valid = np.argwhere(radii_ >= min_radius)[:,0]
        if location == LineLocation.W:
            valid2 = np.argwhere(centers_[0,:] >= 0)[:,0]
            valid = list(set(valid) & set(valid2))
        elif location == LineLocation.E:
            valid2 = np.argwhere(centers_[0,:] < w)[:,0]
            valid = list(set(valid) & set(valid2))
            
        else:
            valid = list(valid)

        efective_circles += len(valid)

        #print "efective_circles",efective_circles
        
        centers = np.c_[centers,centers_[:,valid]]
        radii = np.r_[radii,radii_[valid]]
        
        
    #test
    if False:
        for i in xrange(iterations):
            base1 = base[:,:,i].T
            c2 = Circle.from_base(base1)

            #print c2.centre[0],c2.centre[1],c2.rad,centers[0,i],centers[1,i],radii[i]
            

            
            #assert abs(c2.centre[0] - centers[0,i]) <= 1, "{0} {1}".format(c2.centre[0] , centers[0,i])
            #assert abs(c2.centre[1] - centers[1,i]) <= 1, "{0} {1}".format(c2.centre[1] , centers[1,i])
            #assert abs(c2.rad - radii[i]) <= 5, "{0} {1}".format(c2.rad , radii[i])
    
        #quit()
    
    r2 = radii**2
    
    best_q = 0  # quality(im, centre, rad);
    best_circle = Circle()
    
    k = 0
    k1 = 0
    k2 = 0
    k3 = 0

    for i in xrange(iterations):
        #circle = Circle(centre=centers[:,i],rad=radii[i])
        rad = radii[i]

        q = quality(im, centers[:,i],rad,r2[i],location)

        if debug: print "iter", i, q, best_q

        if q > best_q:
            best_q = q
            best_circle = Circle(centre=centers[:,i],rad=radii[i])

   
    #print iterations,k
    return best_circle


def fit_circle_loop(im, location, iterations, debug=False):

    h, w = im.shape

    if debug: print "image shape",im.shape,"location",location

    vertical_sampling = location == LineLocation.W or location == LineLocation.E

    if vertical_sampling:  # W E
        min_radius = 1.5 * h
        y = np.array([h / 6., h / 2., 5 * h / 6.])
        mu = (w / 2.) * np.ones(3)
        sigma = .5 * w

    else:  # N S
        min_radius = 1.5 * w
        x = np.array([w / 6., w / 2., 5 * w / 6.])
        mu = (h / 2.) * np.ones(3)
        sigma = .5 * h

    best_q = 0  # quality(im, centre, rad);
    best_circle = Circle()

    # Find a global initialisation


    # batch_size = 1000
    # bases = np.zeros((3, 2, batch_size))
    # if vertical_sampling:  # W E
    #     bases[:, 0, :] = w * np.random.rand(3, batch_size)
    # else:  # N S
    #     bases[:, 1, :] = y * np.random.rand(3, batch_size)
    #
    # bases = np.random.random(size=(3, 2, batch_size))
    #
    # centres, radii = circle_vectorized(bases)

    i = 0
    k = 0

    k1 = 0
    k2 = 0
    k3 = 0

    while True:

        if vertical_sampling:  # W E
            x = w * np.random.rand(3)
        else:  # N S
            y = h * np.random.rand(3)

        # points are columns
        base = np.array((x, y))

        circle = Circle.from_base(base)
        rad = circle.radius

        k += 1


        if rad < min_radius:
            k1 += 1
            continue

        if location == LineLocation.W and circle.centre[0] < 0:
            k2 += 1
            continue

        if location == LineLocation.E and circle.centre[0] > w:
            k3 += 1
            continue

        #print im.dtype
        
        #q = quality_cy(im,float(circle.centre[0]),float(circle.centre[1]),float(circle.rad),location)

        q = quality(im, circle.centre,rad,rad**2,location)

        if debug: print "iter", i, q, best_q

        if q > best_q:
            best_q = q
            best_circle = circle

        if i > iterations:
            break

        i += 1

    print k,iterations
    print k1,iterations
    print k2,iterations
    print k3,iterations

    assert best_circle.radius > 0
    return best_circle
    # local refinement: kind of simulated annealing

    if vertical_sampling:  # W E
        mu = x
    else:  # % N S
        mu = y

    i = 0
    sigma_loc = sigma
    while True:

        dim_idx = i % 3
        if vertical_sampling:  # W E
            x[dim_idx] = mu[dim_idx] + np.random.rand() * np.sqrt(sigma)
        else:  # N S
            y[dim_idx] = mu[dim_idx] + np.random.rand() * np.sqrt(sigma)

        base = np.array((x, y))

        circle = Circle.from_base(base)
        rad = circle.radius

        if rad < min_radius:
            continue

        sigma -= .01

        q = quality(im, circle, location)

        # Upgrade quality and sampling parametres
        if q > best_q:
            best_q = q
            best_circle = circle

            # upgrade mu and restart sigma
            if vertical_sampling:  # W E
                mu = x
            else:  # % N S
                mu = y

            sigma = sigma_loc

        if sigma < .01:
            break

        i += 1

    return best_circle

#fit_circle = fit_circle_loop
fit_circle = fit_circle_vectorized

def circinter(R, d, r):
    # R -- Radius first circle
    # r -- Radius second circle
    # d -- distance between centres of both circles

    assert r > 0
    if d <= epsilon:
        return np.pi

    x = (d*d-r*r+R*R)/(2.*d)
    rat = x/R
    if rat <= -1.0:
        return np.pi

    return np.arccos(rat)


def find_corner(circ1, circ2, ref):
    """Find a corner defined by the intersection of two circles"""

    rad1 = circ1.radius
    d = norm(circ2.centre-circ1.centre)
    rad2 = circ2.radius

    ang = circinter(rad1, d, rad2)

    pc = rad1*((circ2.centre-circ1.centre)/d)

    rot1 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    rot2 = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])

    p1 = circ1.centre + np.dot(rot1, pc.T)
    p2 = circ1.centre + np.dot(rot2, pc.T)

    d1 = norm(p1 - ref)
    d2 = norm(p2 - ref)

    if d1 < d2:
        return p1
    else:
        return p2


def circ_eval_x(x, circle, h):

    a = np.sqrt(circle.radius**2 - (x - circle.centre[0])**2)

    y1 = circle.centre[1] + a
    v1 = np.sum(np.logical_and(y1 >= 0, y1 < h))

    y2 = circle.centre[1] - a
    v2 = np.sum(np.logical_and(y2 >= 0, y2 < h))

    if v1 > v2:
        return y1
    else:
        return y2


def circ_eval_y(x, circle, w):

    a = np.sqrt(circle.radius**2 - (x - circle.centre[1])**2)

    x1 = circle.centre[0] + a
    v1 = np.sum(np.logical_and(x1 >= 0, x1 < w))

    x2 = circle.centre[0] - a
    v2 = np.sum(np.logical_and(x2 >= 0, x2 < w))

    if v1 > v2:
        return x1
    else:
        return x2


def find_circles(im, iterations, line_offset=4):

    h, w = im.shape
    divs = 4.

    im_e = im[:, int(((divs-1) / divs) * w):]
    im_w = im[:, 1:int((1 / divs) * w)]
    im_n = im[:int((1 / divs) * h), :]
    im_s = im[int(((divs-1) / divs) * h):, :]

    e = fit_circle(im_e, LineLocation.E,iterations)
    e.centre += np.array([int(((divs-1) / divs) * w), 0], np.int)
    e.centre[0] += line_offset

    w = fit_circle(im_w, LineLocation.W,iterations)
    w.centre[0] -= line_offset

    n = fit_circle(im_n, LineLocation.N,iterations)
    n.centre[1] -= line_offset

    s = fit_circle(im_s, LineLocation.S,iterations)
    s.centre += np.array([0, int(((divs-1) / divs) * h)])
    s.centre[1] += line_offset

    return n, s, w, e


def find_corners(im, circle_n, circle_s, circle_w, circle_e):

    h, w = im.shape

    a = find_corner(circle_w, circle_n, np.array([0, 0]))
    b = find_corner(circle_e, circle_n, np.array([w - 1, 0]))
    c = find_corner(circle_e, circle_s, np.array([w - 1, h - 1]))
    d = find_corner(circle_w, circle_s, np.array([0, h - 1]))

    return a, b, c, d


def rectify(original, iterations=5000, ds=7, pad=4):
    # Rectify a distortiend image (original) to a rectangle. Each edge is discretised in ds points
    # The rectified size can be target_size if is not None. If it is None same size as original

    # detect edges
    im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (0, 0), 1.5)
    im = auto_canny(im)

    # avoid artifacts in the edge of the image
    im[-pad:, :] = 0
    im[:pad, :] = 0
    im[:, :pad] = 0
    im[:, -pad:] = 0

    h, w = im.shape

    circle_n, circle_s, circle_w, circle_e = find_circles(im,iterations=iterations)

    a, b, c, d = find_corners(im, circle_n, circle_s, circle_w, circle_e)

    # Obtain correspondences
    ptsN = np.zeros([ds, 2])
    ptsS = np.zeros([ds, 2])
    ptsW = np.zeros([ds, 2])
    ptsE = np.zeros([ds, 2])

    ptsN[:, 0] = np.linspace(a[0], b[0], ds)
    ptsN[:, 1] = circ_eval_x(ptsN[:, 0], circle_n, h)

    ptsS[:, 0] = np.linspace(d[0], c[0], ds)
    ptsS[:, 1] = circ_eval_x(ptsS[:, 0], circle_s, h)

    ptsW[:, 1] = np.linspace(a[1], d[1], ds)
    ptsW[:, 0] = circ_eval_y(ptsW[:, 1], circle_w, w)

    ptsE[:, 1] = np.linspace(b[1], c[1], ds)
    ptsE[:, 0] = circ_eval_y(ptsE[:, 1], circle_e, w)

    # N S W E

    xN = np.linspace(0, w-1, ds)
    xS = np.linspace(0, w-1, ds)

    yW = np.linspace(0, h-1, ds)
    yE = np.linspace(0, h-1, ds)

    pd = np.array([xN[1:-1], np.zeros(ds-2)]).T
    pd = np.concatenate((pd, np.array([xS[1:-1], (h-1) * np.ones(ds-2)]).T))
    pd = np.concatenate((pd, np.array([np.zeros(ds), yW]).T))
    pd = np.concatenate((pd, np.array([(w-1) * np.ones(ds), yE]).T))

    ps = np.concatenate([ptsN[1:-1, :], ptsS[1:-1], ptsW, ptsE])

    # Thin plate warping
    rectified = rbfwarp2d(original, ps, pd, method="g", r=.5 * w)

    circles = {'north': circle_n, 'south': circle_s, 'west': circle_w, 'east': circle_e}
    matches = {'north': ptsN, 'south': ptsS, 'west': ptsW, 'east': ptsE}
    return rectified, circles, matches


def main():

    import matplotlib.pyplot as plt

    def plot_circle(circle):
        artist = plt.Circle(circle.centre, circle.radius, color='g', clip_on=True, fill=False)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(artist)


    np.random.seed(2)

    # image_filename = '../matlab/im.tiff'
    image_filename = 'frame-54.tiff'
    original = cv2.imread(image_filename)
    im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    h, w = im.shape

    # imCanny = im
    # imCanny = cv2.GaussianBlur(imCanny, (0, 0), 1.5)
    # imCanny = auto_canny(imCanny)
    # plt.imshow(imCanny, cmap='Greys_r')
    # plt.show()
    # quit()

    plt.subplot(121)
    rectified, circles, matches = rectify(original)
    plt.imshow(im, cmap='Greys_r')

    plot_circle(circles['north'])
    plot_circle(circles['west'])
    plot_circle(circles['south'])
    plot_circle(circles['east'])

    plt.plot(matches['north'][:, 0], matches['north'][:, 1], 'oy')
    plt.plot(matches['south'][:, 0], matches['south'][:, 1], 'oy')
    plt.plot(matches['west'][:, 0],  matches['west'][:, 1], 'oy')
    plt.plot(matches['east'][:, 0],  matches['east'][:, 1], 'oy')

    plt.subplot(122)
    plt.imshow(rectified)

    plt.show()
    quit()

if __name__ == "__main__":
    main()
