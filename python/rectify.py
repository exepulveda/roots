# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:56:23 2016

@author: aparra
"""

import numpy as np
import cv2
from rbfwarp2d import rbfwarp2d

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

        centre = np.array([0, 0])

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


class LineLocation:
    N, S, W, E = range(4)


def improfile(im, x, y):
    # remove nan values and cast to int
    idx = ~np.isnan(x) & ~np.isnan(y)
    x = x[idx].round().astype(np.int)
    y = y[idx].round().astype(np.int)

    # remove out of the image points
    h, w = im.shape
    idx = (x >= 0) * (x < w) * (y >= 0) * (y < h)
    x = x[idx]
    y = y[idx]

    val = im[y, x].astype(np.double)

    if val.shape[0] == 0:
        return 0

    val = sum(val) / (np.max([w, h, val.shape[0]]))
    return val


def improfile_north_line(im, x, y):
    # remove nan values and cast to int
    y_l = y + 5

    idx = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(y_l)
    x = x[idx].round().astype(np.int)
    y = y[idx].round().astype(np.int)
    y_l = y_l[idx].round().astype(np.int)

    # remove out of the image points
    h, w = im.shape
    idx = (x >= 0) * (x < w) * (y >= 0) * (y < h) * (y_l >= 0) * (y_l < h)
    x = x[idx]
    y = y[idx]
    y_l = y_l[idx]

    val = im[y, x].astype(np.double)
    val_l = im[y_l, x].astype(np.double)
    val = val-val_l
    # neg_values = val <= 0
    # val **= 2
    # val[neg_values] = -val[neg_values]

    if val.shape[0] == 0:
        return 0

    val = sum(val) / (np.max([w, h, val.shape[0]]))
    return val


def quality(im, circle, location):
    h, w = im.shape

    centre = circle.centre
    r = circle.radius

    if location == LineLocation.W or location == LineLocation.E:
        yp = np.arange(h)
        xp = centre[0] + np.sqrt(r**2 - (yp - centre[1])**2)

        val1 = improfile(im, xp, yp)

        xp = centre[0] - np.sqrt(r**2 - (yp - centre[1])**2)
        val2 = improfile(im, xp, yp)

    elif location == LineLocation.N:
        xp = np.arange(w)
        yp = centre[1] + np.sqrt(r**2 - (xp - centre[0])**2)

        val1 = improfile_north_line(im, xp, yp)

        yp = centre[1] - np.sqrt(r**2 - (xp - centre[0])**2)
        val2 = improfile_north_line(im, xp, yp)

    else:  # S
        xp = np.arange(w)
        yp = centre[1] + np.sqrt(r ** 2 - (xp - centre[0]) ** 2)

        val1 = improfile(im, xp, yp)

        yp = centre[1] - np.sqrt(r ** 2 - (xp - centre[0]) ** 2)
        val2 = improfile(im, xp, yp)

    return max(val1, val2)


def fit_circle(im, location):
    # np.random.seed(2)

    im = cv2.GaussianBlur(im, (0, 0), 3)
    h, w = im.shape

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
    i = 0
    while True:

        if vertical_sampling:  # W E
            x = w * np.random.rand(3)
        else:  # N S
            y = h * np.random.rand(3)

        base = np.array((x, y))

        circle = Circle.from_base(base)
        rad = circle.radius

        if rad < min_radius:
            continue

        q = quality(im, circle, location)
        if q > best_q:
            best_q = q
            best_circle = circle

        if i > 1500:
            break

        i += 1

    # return best_circle
    # local refinement: kind of simulated annealing

    if vertical_sampling:  # W E
        mu = x
    else:  # % N S
        mu = y

    i = 0
    sigma_loc = .1*sigma
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

        sigma *= .99

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


def circinter(R, d, r):
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


def find_circles(im):
    h, w = im.shape

    im_e = im[:, int((2 / 3.) * w):]
    im_w = im[:, 1:int((1 / 3.) * w)]
    im_n = im[:int((1 / 3.) * h), :]
    im_s = im[int((2 / 3.) * h):, :]

    e = fit_circle(im_e, LineLocation.E)
    e.centre += np.array([int((2 / 3.) * w), 0], np.int)

    w = fit_circle(im_w, LineLocation.W)

    n = fit_circle(im_n, LineLocation.N)

    s = fit_circle(im_s, LineLocation.S)
    s.centre += np.array([0, int((2 / 3.) * h)])

    return n, s, w, e


def find_corners(im, circle_n, circle_s, circle_w, circle_e):
    h, w = im.shape

    a = find_corner(circle_w, circle_n, np.array([0, 0]))
    b = find_corner(circle_e, circle_n, np.array([w - 1, 0]))
    c = find_corner(circle_e, circle_s, np.array([w - 1, h - 1]))
    d = find_corner(circle_w, circle_s, np.array([0, h - 1]))

    return a, b, c, d


def rectify(original, ds=7):
    # Rectify a distortiend image (original) to a rectangle. Each edge is discretised in ds points
    # The rectified size can be target_size if is not None. If it is None same size as original

    im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    h, w = im.shape

    circle_n, circle_s, circle_w, circle_e = find_circles(im)

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

    # image_filename = '../matlab/im.tiff'
    image_filename = 'frame-52.tiff'
    original = cv2.imread(image_filename)
    im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    rectified, circles, matches = rectify(original)

    plt.subplot(121)
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
