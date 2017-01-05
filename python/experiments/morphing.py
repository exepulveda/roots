"""
/**
 *  Thin Plate Spline 2D point morpher example.
 *
 *    Takes in sample point coordinates and their displacements,
 *    fits a TPS into them and allowes morphing new points
 *    accordint to the approximated deformation.
 *
 *    Supports TPS approximation 3 as suggested in paper
 *    Gianluca Donato and Serge Belongie, 2002: "Approximation
 *    Methods for Thin Plate Spline Mappings and Principal Warps"
 *
 *  This should be considered more as an example than a ready made module!
 *  The code has been extracted from a working "deformed shape matching"
 *  application and has been optimized for that particular case.
 *  I don't even know if this compiles or not.
 *
 *  Copyright (C) 2003-2005 by Jarno Elonen
 *
 *  This is Free Software / Open Source with a very permissive
 *  license:
 *
 *  Permission to use, copy, modify, distribute and sell this software
 *  and its documentation for any purpose is hereby granted without fee,
 *  provided that the above copyright notice appear in all copies and
 *  that both that copyright notice and this permission notice appear
 *  in supporting documentation.  The authors make no representations
 *  about the suitability of this software for any purpose.
 *  It is provided "as is" without express or implied warranty.
 */
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

def base_func_2(r2):
    ret = np.zeros_like(r2)

    #find 0
    indices = np.where(r2 != 0)
    ret[indices] = r2[indices] * np.log(r2[indices]) * 0.217147241
    
    return ret
    
def base_func(d):    
    return np.exp(-d/(10*200)**2)


#/// 2D point morph interpolator using TPS (Thin Plate Spline).
class Morpher(object):
    def __init__(self,source_points, destination_points,regularization,subsampling_factor=1.0):
        assert source_points.shape == destination_points.shape
        
        p,n = source_points.shape
        
        m = int(p * subsampling_factor)
        if ( m < 3 ): m = 3
        if ( m > p ): m = p

        if m < p:
            pass
            #// Randomize the input if subsampling is used
            #for ( unsigned i=0; i<samples->size()-1; ++i )
            #{
            #int j = i + ((unsigned)rand()) % (samples->size()-i);
            #Coord_Diff tmp = (*samples)[j];
            #(*samples)[i] = (*samples)[j];
            #(*samples)[j] = tmp;
            #}

        #// Allocate the matrix and vector
        mtx_l = np.zeros((p+3, m+3))
        mtx_v = np.zeros((p+3, 2))
        mtx_orig_k = np.zeros((p, m))

        for i in xrange(m):
            dx = source_points[i,:]-destination_points
            mtx_orig_k[:,i] = np.sum(dx**2,axis=1)


        #distance of differences
        diff_points = (source_points-destination_points)
        dists = pdist(diff_points)
        dists = squareform(dists)
        elen2 = base_func(dists**2)
        
        a = np.sum(dists)
        a /= (p**2)
        
        print "a",a
        
        mtx_orig_k = base_func(mtx_orig_k)
        
        np.fill_diagonal(mtx_orig_k,a)

        
        mtx_l[:p,:m] = mtx_orig_k[:p,:m]


        #Fill K (p x m, upper left of L)
        #a = 0.0
        #for i in xrange(p):
        #    ps_i = source_points[i,:]
        #    pd_i = destination_points[i,:]
        #    for j in xrange(m):
        #        ps_j = source_points[j,:]
        #        pd_j = destination_points[j,:]
                
        #        elen2 = (ps_i[0]-ps_j[0])**2 + (ps_i[1]-ps_j[1])**2
            
        #        mtx_orig_k[i,j] = base_func(elen2)
        #        mtx_l[i,j] = mtx_orig_k[i,j]
                
        #        a += elen2 * 2
                
        #a /= (p*p) 
        
        #// Empiric value for avg. distance between points
        #//
        #// This variable is normally calculated to make regularization
        #// scale independent, but since our shapes in this application are always
        #// normalized to maxspect [-.5,.5]x[-.5,.5], this approximation is pretty
        #// safe and saves us p*p square roots
        #a = 0.5

        #// Fill the rest of L
        
        #P (p x 3, upper right)
        mtx_l[:p,m] = 1.0
        mtx_l[:p,m+1:] = source_points[:]
        mtx_l[m,:p] = 1.0
        mtx_l[m+1:m+3,:p] = source_points[:].T
        
        #for i in xrange(p):
        #    ps_i = source_points[i]
        #    pd_i = destination_points[i]

        #    #P (p x 3, upper right)
        #    mtx_l[i, m+0] = 1.0
        #    mtx_l[i, m+1] = ps_i[0]
        #    mtx_l[i, m+2] = ps_i[1]

        #    if i<m:
        #        #diagonal: reqularization parameters (lambda * a^2)
        #        mtx_orig_k[i,i] = regularization * (a*a)
        #        mtx_l[i,i] = mtx_orig_k[i,i]

        #        #transposed (3 x p, bottom left)
        #        mtx_l[p+0, i] = 1.0;
        #        mtx_l[p+1, i] = pd_i[0]
        #        mtx_l[p+2, i] = pd_i[1]

        # O (3 x 3, lower right)
        #for ( unsigned i=p; i<p+3; ++i )
        #  for ( unsigned j=m; j<m+3; ++j )
        #    mtx_l(i,j) = 0.0;

        np.savetxt("L.csv",mtx_l)

        #Fill the right hand matrix V
        mtx_v[:p,:] = diff_points
        
        #for i in xrange(p):
        #    ps_i = source_points[i]
        #    pd_i = destination_points[i]
        #    mtx_v[i,0] = (ps_i[0] - pd_i[0])
        #    mtx_v[i,1] = (ps_i[1] - pd_i[1])

        #mtx_v[p+0, 0) = mtx_v(p+1, 0) = mtx_v(p+2, 0) = 0.0;
        #mtx_v[p+0, 1) = mtx_v(p+1, 1) = mtx_v(p+2, 1) = 0.0;

        #// Solve the linear system "inplace"
        x = np.linalg.solve(mtx_l, mtx_v)
        mtx_w = x
        
        #store
        self.mtx_orig_k = mtx_orig_k
        self.mtx_v = mtx_v
        self.mtx_w = mtx_w
        self.source_points = source_points
        self.destination_points = destination_points
        self.diff = diff_points
        
    def interpolate(self,image):
        h,w = image.shape[0:2]
        print "h,w",h,w
        
        [x,y] = np.meshgrid(np.arange(w),np.arange(h))
        
        x = np.arange(w)
        y = np.zeros(w)
        
        #x = x.flatten()
        #y = y.flatten()
        
        print "morphing coordinates"
        
        mor = self.morph(np.c_[x,y])
        
        #reshape to 2d image
        
        xd = mor[:,0]
        yd = mor[:,1]
        
        for i in xrange(w):
            print i,x[i],y[i],xd[i],yd[i]
        
        plt.plot(xd,yd,"x-r")
        plt.show()

        np.savetxt("xy.csv",np.c_[x,y])
        np.savetxt("mor.csv",mor)

        #create an interpolator
        print "creating 2d interpolator"
        interpoler= RectBivariateSpline(np.arange(h),np.arange(w),image)
        
        print "interpolating"
        imt = interpoler(xd,yd,grid=False)
        
        return imt.reshape((w,h)) .T
        

    def morph(self,points):
        #Morph given points according to the TPS
        #@param pts The points to morph

        #build meshgrid 

        m = len(self.mtx_orig_k)
        n = len(points)

        Kp = np.empty((n,m))
        
        for k in xrange(m):
            dx = self.source_points[k,:]-points
            Kp[:,k] = np.sum(dx**2,axis=1)

        Kp = base_func(Kp)
        
        L = np.c_[Kp,np.ones(n),points]
        
        print "L",L.shape,"w",self.mtx_w.shape
        
        ptall = np.dot(L,self.mtx_w)
        
        return ptall
