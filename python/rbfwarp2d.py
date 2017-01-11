import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.linalg
import cv2


def homo(im, ps, pd):
    h, status = cv2.findHomography(ps, pd)
    #print h, status
    im_out = cv2.warpPerspective(im, h, (im.shape[1], im.shape[0]))
    # im_dst = cv2.warpPerspective(im, h, size)
    return im_out


def wrap(im, ps, pd):
    return tf.warp(im, ps, pd)


def rbfwarp2d(im, ps, pd, method="g", r=None):
    """function [imo,mask] = rbfwarp2d( im, ps, pd, varargin )
    % Radial base function/Thin-plate spline 2D image warping.
    % [imo,mask] = rbfwarp2d( im, ps, pd, method)
    %   input:
    %       im: image 2d matrix
    %       ps: 2d source landmark [n*2]
    %       pd: 2d destin landmark [n*2]
    %       method:
    %         'gau',r  - for Gaussian function   ko = exp(-|pi-pj|/r.^2);
    %         'thin'   - for Thin plate function ko = (|pi-pj|^2) * log(|pi-pj|^2)
    %   output:
    %       imo  : output matrix
    %       mask : mask for output matrix, 0/1 means out/in the border
    %
    %   Bookstein, F. L. 
    %   "Principal Warps: Thin Plate Splines and the Decomposition of Deformations."
    %   IEEE Trans. Pattern Anal. Mach. Intell. 11, 567-585, 1989. 
    %
    %   Code by WangLin
    %   2015-11-5
    %   wanglin193@hotmail.com
    """

    # initialize default parameters
    if len(im.shape) >= 3:
        imh, imw, imc = im.shape
    else:
        imh, imw = im.shape
        imc = 1

    if r is None:
        r = 0.1 * w

    # imo = np.zeros((imh, imw, imc))
    # mask = np.zeros((imh,imw))

    # %% Training w with L
    nump = len(pd)
    num_center = len(ps)
    K = np.zeros((nump, num_center))

    for i in xrange(num_center):
        # Inverse warping from destination!
        # dx = np.ones(nump,1)*ps(i,:)-pd;
        dx = np.empty((nump, 2))
        dx[:, 0] = ps[i, 0]
        dx[:, 1] = ps[i, 1]
        dx -= pd

        K[:, i] = np.sum(dx ** 2, axis=1)

    if method == "g":
        K = rbf(K, r)
    elif method == "t":
        K = ThinPlate(K)

    # % P = [1,xp,yp] where (xp,yp) are n landmark points (nx2)
    # P[0, = [ones(num_center,1),pd];

    P = np.ones((num_center, 3))
    P[:, 1:] = pd[:, :]

    L = np.bmat([[K, P], [P.T, np.zeros((3, 3))]])

    # Y = [x,y;
    #      0,0]; (n+3)x2
    Y = np.r_[ps, np.zeros((3, 2))]
    w = scipy.linalg.solve(L, Y)

    # Using w
    [x, y] = np.meshgrid(np.arange(imw) + 1, np.arange(imh) + 1)
    # pt = [x(:), y(:)]; #column concat

    pt = np.c_[x.flatten(order='F'), y.flatten(order='F')]

    nump = len(pt)
    Kp = np.zeros((nump, num_center))
    for i in xrange(num_center):
        # dx = np.linalg.dot(np.ones(nump),ps[i,:])-pt  #(56,2)
        dx = np.empty((nump, 2))
        dx[:, 0] = ps[i, 0]
        dx[:, 1] = ps[i, 1]
        dx = dx - pt
        Kp[:, i] = np.sum(dx ** 2, axis=1)

    if method == "g":
        Kp = rbf(Kp, r)
    elif method == "t":
        Kp = ThinPlate(Kp)

    L2 = np.c_[Kp, np.ones(nump), pt]

    ptall = np.dot(L2, w)

    # reshape to 2d image
    xd = ptall[:, 0].reshape((imh, imw), order="F")
    yd = ptall[:, 1].reshape((imh, imw), order="F")

    if imc > 1:
        rect = np.empty((imh, imw, imc))
        for i in xrange(imc):
            interpoler = RectBivariateSpline(np.arange(imh), np.arange(imw), im[:, :, i])
            imt = interpoler(yd, xd, grid=False)
            rect[:, :, i] = imt.reshape((imh, imw))

    else:
        interpoler = RectBivariateSpline(np.arange(imh), np.arange(imw), im)
        imt = interpoler(yd, xd, grid=False)
        rect = imt.reshape((imh, imw))

    rect = cv2.convertScaleAbs(rect)
    #cv2.xphoto.balanceWhite(rect, rect, cv2.xphoto.WHITE_BALANCE_SIMPLE)
    return rect


def rbf(d, r):
    return np.exp(-d / (r ** 2))


def ThinPlate(ri):
    r1i = ri
    # r1i((ri==0))=realmin; % Avoid log(0)=inf
    return ri * np.log(r1i)
