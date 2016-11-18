import numpy as np

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d

import scipy.linalg

from skimage import transform as tf

import cv2

def homo(im,ps,pd):
    h, status = cv2.findHomography(ps, pd)
    print h,status
    im_out = cv2.warpPerspective(im, h, (im.shape[1],im.shape[0]))
    #im_dst = cv2.warpPerspective(im, h, size)
    return im_out
    
def wrap(im,ps,pd):
    
    return tf.warp(im,ps,pd)

def rbfwarp2d(im, ps, pd, method = "g",r=None):
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
        imh,imw,imc = im.shape
    else:
        imh,imw = im.shape
        imc = 1
        
    if r is None:
        r = 0.1*imw
        
    print imh,imw,imc
    
    imo = np.zeros((imh,imw,imc))
    mask = np.zeros((imh,imw))

    #%% Training w with L
    nump = len(pd)
    num_center = len(ps)
    K=np.zeros((nump,num_center))

    for i in xrange(num_center):
        #Inverse warping from destination!
        #dx = np.ones(nump,1)*ps(i,:)-pd; 
        dx = np.empty((nump,2))
        dx[:,0] = ps[i,0]
        dx[:,1] = ps[i,1]
        dx = dx - pd


        K[:,i] = np.sum(dx**2,axis=1)

    if method == "g":
        K = rbf(K,r)
    elif method == "t":
        K = ThinPlate(K)



    #% P = [1,xp,yp] where (xp,yp) are n landmark points (nx2)
    #P[0, = [ones(num_center,1),pd];
    P = np.ones((nump,3))
    P[:,1:] = pd[:,:]
    
    #for i in xrange(nump):
    #    print i,P[i,:]
        
    #quit()
    #print "P",P
    # L = [ K  P;
    #       P' 0 ]
    #L = [K,P;P',zeros(3,3)];
    
    L = np.bmat([[K,P] ,[P.T,np.zeros((3,3))]])
    
    #for i in xrange(len(L)):
    #    print i,L[i,:]
    #print L.shape
    #print "L",L
    #quit()


    
    # Y = [x,y;
    #      0,0]; (n+3)x2
    Y = np.r_[ps,np.zeros((3,2))]
    #Y = np.bmat([[ps,P] ,[P.T,np.zeros((3,3))]])
    #w = np.inv(L)*Y;
    #for i in xrange(len(Y)):
    #    print i,Y[i,:]

    np.savetxt("L.csv",L)
    np.savetxt("Y.csv",Y)


    #w = np.linalg.solve(L,Y)
    w = scipy.linalg.solve(L, Y)
    
    
    #w = np.dot(np.linalg.inv(L),Y)



    #%% Using w
    [x,y] = np.meshgrid(np.arange(imw)+1,np.arange(imh)+1)
    #pt = [x(:), y(:)]; #column concat

    

    pt = np.c_[x.flatten(order='F'),y.flatten(order='F')]
    



    nump = len(pt)
    Kp = np.zeros((nump,num_center))
    for i in xrange(num_center):
        #dx = np.linalg.dot(np.ones(nump),ps[i,:])-pt  #(56,2)
        dx = np.empty((nump,2))
        dx[:,0] = ps[i,0]
        dx[:,1] = ps[i,1]
        dx = dx - pt
        Kp[:,i] = np.sum(dx**2,axis=1)

    if method == "g":
        Kp = rbf(Kp,r)
    elif method == "t":
        Kp = ThinPlate(Kp)

    print "method",method
    np.savetxt("Kp.csv",Kp)


    L2 = np.c_[Kp,np.ones(nump),pt]

    ptall = np.dot(L2,w)


    #reshape to 2d image
    xd = ptall[:,0].reshape((imh,imw),order="F")
    yd = ptall[:,1].reshape((imh,imw),order="F")
    
    if imc > 1:
        rect = np.empty((imh,imw,imc))
        for i in xrange(imc):
            interpoler= RectBivariateSpline(np.arange(imw),np.arange(imh),im[:,:,i])
            imt = interpoler(xd,yd,grid=False)

            rect[:,:,i] = imt.reshape((imh,imw),order="C")

    else:
        rect = np.empty((imw,imh))
        print imh,imw,im.shape
        print "creating interp2d"
        #f = interp2d(np.arange(imw),np.arange(imh),im[:,:],copy=False)
        
        #print "evaluating interp2d"
        #imt = f(xd,yd)
        
        im = np.float32(im)
        
        interpoler= RectBivariateSpline(np.arange(imh)+1,np.arange(imw)+1,im)

        ret = np.empty(imh*imw)

        i = 0
        for j in xrange(imh):
            for k in xrange(imw):
                imt = interpoler(yd[j,k],xd[j,k],grid=False)
                #print j+1,k+1,yd[j,k],xd[j,k],imt,im[j,k]
                ret[i] = imt
                i=i+1


        rect = ret.reshape((imh,imw),order="F")
        #imt = interpoler(xd,yd,grid=False)
        #for i in xrange(10):
        #    imt = interpoler(xd[i],yd[i],grid=False)
        #    print i,xd[i],yd[i],imt

        #quit()
        #print interpoler(8.0,18.0,grid=False)
        #print interpoler(18.0,8.0,grid=False)
        #quit()
        #imt = interp2( single(im(:,:,i)),xd,yd,'linear');
        #f = interpolate.interp2d(x, y, z, kind='cubic')
        
        #imo[:,:,i] = imt
        #rect = imt.reshape((imh,imw),order="F")
    
        
    return rect
    

def rbf(d,r):
    return np.exp(-d/(r**2))

def ThinPlate(ri):
    r1i = ri
    #r1i((ri==0))=realmin; % Avoid log(0)=inf
    return ri*np.log(r1i)
