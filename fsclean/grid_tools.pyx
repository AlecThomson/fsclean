"""
grid_tools.pyx

3D gridding routines for use with fsclean

**********************************************************************************

Copyright 2012 Michael Bell

This file is part of fsclean.

fsclean is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

fsclean is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with fsclean.  If not, see <http://www.gnu.org/licenses/>.

**********************************************************************************

"""

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
CTYPE = np.complex128
ctypedef np.float64_t DTYPE_t 
ctypedef np.complex128_t CTYPE_t

cdef extern from "gsl/gsl_sf_bessel.h":
    double gsl_sf_bessel_I0(double x)

cdef extern from "math.h":
    double exp(double theta)
    double sqrt(double x)
    double ceil(double x)
    double sin(double theta)
    
@cython.boundscheck(False)
@cython.wraparound(False)

def grid_3d_pol(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v,\
    double l2, \
    np.ndarray[CTYPE_t, ndim=1] Qvis, np.ndarray[CTYPE_t, ndim=1] Uvis, \
    double du, double dv, double dl2, double alpha, int W):
        
        cdef int W3 = W**3
        
        cdef int nvis = u.shape[0]

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ug = \
            np.zeros(nvis*W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vg = \
            np.zeros(nvis*W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] l2g = \
            np.zeros(nvis*W3, dtype=DTYPE)

        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Qvisg = \
            np.zeros(nvis*W3, dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Uvisg = \
            np.zeros(nvis*W3, dtype=CTYPE)
        
        # holds the W values after u gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu1 = \
            np.zeros(W, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tQvis1 = \
            np.zeros(W, dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tUvis1 = \
            np.zeros(W, dtype=CTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv1 = \
            np.zeros(W, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tl21 = \
            np.zeros(W, dtype=DTYPE)

        
        # holds the W**2 values after subsequent v gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu2 =\
            np.zeros(W**2, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tQvis2 = \
            np.zeros(W**2, dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tUvis2 = \
            np.zeros(W**2, dtype=CTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv2 = \
            np.zeros(W**2, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tl22 = \
            np.zeros(W**2, dtype=DTYPE)
        
        
        # holds the W**3 values after subsequent l2 gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu3 = \
            np.zeros(W3, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tQvis3 = \
            np.zeros(W3, dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tUvis3 = \
            np.zeros(W3, dtype=CTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv3 = \
            np.zeros(W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tl23 = \
            np.zeros(W3, dtype=DTYPE)
        
        
        cdef Py_ssize_t i
        
        # From Beatty et al. (2005)
        cdef double beta = np.pi*np.sqrt((W/alpha)**2.*(alpha - 0.5)**2 - 0.8)
        
        for i in range(nvis):
            
            # For each visibility point, grid in 3D, one dimension at a time
            # so each visibility becomes W**3 values located on the grid
            
            # Grid in u
            grid_1d_complex_scalar(u[i], Qvis[i], Uvis[i], du, W, \
                beta, v[i], l2, \
                tu1, tQvis1, tUvis1, tv1, tl21) # output arrays
                     
            
            # Grid in v
            grid_1d_complex(tv1, tQvis1, tUvis1, dv, W, beta, tu1, tl21, \
                tv2, tQvis2, tUvis2, tu2, tl22) # output arrays
                    
            
            # Grid in l2
            grid_1d_complex(tl22, tQvis2, tUvis2, dl2, W, beta, tu2, tv2, \
                tl23, tQvis3, tUvis3, tu3, tv3) # output arrays
                    
            
            ug[i*W3:(i+1)*W3] = tu3
            vg[i*W3:(i+1)*W3] = tv3
            l2g[i*W3:(i+1)*W3] = tl23

            Qvisg[i*W3:(i+1)*W3] = tQvis3
            Uvisg[i*W3:(i+1)*W3] = tUvis3
        
        return ug, vg, l2g, Qvisg, Uvisg
        
def grid_3d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v,\
    double l2, \
    np.ndarray[DTYPE_t, ndim=1] vis, \
    double du, double dv, double dl2, double alpha, int W):
        
        cdef int W3 = W**3
        
        cdef int nvis = u.shape[0]

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ug = \
            np.zeros(nvis*W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vg = \
            np.zeros(nvis*W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] l2g = \
            np.zeros(nvis*W3, dtype=DTYPE)

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] visg = \
            np.zeros(nvis*W3, dtype=DTYPE)
        
        # holds the W values after u gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu1 = \
            np.zeros(W, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tvis1 = \
            np.zeros(W, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv1 = \
            np.zeros(W, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tl21 = \
            np.zeros(W, dtype=DTYPE)

        
        # holds the W**2 values after subsequent v gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu2 =\
            np.zeros(W**2, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tvis2 = \
            np.zeros(W**2, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv2 = \
            np.zeros(W**2, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tl22 = \
            np.zeros(W**2, dtype=DTYPE)
        
        
        # holds the W**3 values after subsequent l2 gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu3 = \
            np.zeros(W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tvis3 = \
            np.zeros(W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv3 = \
            np.zeros(W3, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tl23 = \
            np.zeros(W3, dtype=DTYPE)
        
        
        cdef Py_ssize_t i
        
        # From Beatty et al. (2005)
        cdef double beta = np.pi*np.sqrt((W/alpha)**2.*(alpha - 0.5)**2 - 0.8)
        
        for i in range(nvis):
            
            # For each visibility point, grid in 3D, one dimension at a time
            # so each visibility becomes W**3 values located on the grid
            
            # Grid in u
            grid_1d_scalar(u[i], vis[i], du, W, \
                beta, v[i], l2, \
                tu1, tvis1, tv1, tl21) # output arrays
                     
            
            # Grid in v
            grid_1d(tv1, tvis1, dv, W, beta, tu1, tl21, \
                tv2, tvis2, tu2, tl22) # output arrays
                    
            
            # Grid in l2
            grid_1d(tl22, tvis2, dl2, W, beta, tu2, tv2, \
                tl23, tvis3, tu3, tv3) # output arrays
                    
            
            ug[i*W3:(i+1)*W3] = tu3
            vg[i*W3:(i+1)*W3] = tv3
            l2g[i*W3:(i+1)*W3] = tl23

            visg[i*W3:(i+1)*W3] = tvis3
        
        return ug, vg, l2g, visg
        
        
        
        
def degrid_3d_complex(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v,\
    double w, np.ndarray[CTYPE_t, ndim=3] regVis, \
    double du, double Nu, double umin, double dv, double Nv, double vmin, \
    double dw, double Nw, double wmin, double alpha, int W):

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
            np.arange(0.,Nu,1.)*du + umin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vgrid = \
            np.arange(0.,Nv,1.)*dv + vmin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] wgrid = \
            np.arange(0.,Nw,1.)*dw + wmin
            
        cdef int nvis = u.shape[0]
    
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Vis = \
            np.zeros(nvis, dtype=CTYPE)
        
        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)
        # Grid in u and v
        cdef double Du = W*du
        cdef double Dv = W*dv
        cdef double Dw = W*dw
        
        cdef Py_ssize_t i, j, urang, vrang, wrang, k, Wu, Wv, l
        
        cdef double gcf_val_u, gcf_val_v, gcf_val_w, gcf_val, upos, vpos
        cdef double delu, delv, delw, temp


        wrang = int(np.ceil((w - 0.5*Dw - wmin)/dw))
    
        for k in range(nvis):
            upos = u[k]
            vpos = v[k]
            
            urang = int(np.ceil((upos - 0.5*Du - umin)/du))
            vrang = int(np.ceil((vpos - 0.5*Dv - vmin)/dv))
            
    
            for i in range(urang, urang+W):
                if i>=Nu or i<0: continue
                delu = upos - ugrid[i]
                gcf_val_u = gcf_kaiser(delu, Du, beta)
                
                for j in range(vrang, vrang+W):
                    if j>=Nv or j<0: continue
                    delv = vpos-vgrid[j]
                    gcf_val_v = gcf_kaiser(delv, Dv, beta)
                    
                    for l in range(wrang, wrang+W):
                        if l>=Nw or l<0: continue
                        delw = w-wgrid[l]
                        gcf_val_w = gcf_kaiser(delw, Dw, beta)
                        
                        gcf_val = gcf_val_u*gcf_val_v*gcf_val_w                                                
                        Vis[k] = Vis[k] + regVis[l,j,i]*gcf_val
    
        return Vis
        
        
        
def degrid_3d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v, \
    double w, np.ndarray[DTYPE_t, ndim=3] regVis, \
    double du, double Nu, double umin, double dv, double Nv, double vmin, \
    double dw, double Nw, double wmin, double alpha, int W):

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
            np.arange(0.,Nu,1.)*du + umin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vgrid = \
            np.arange(0.,Nv,1.)*dv + vmin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] wgrid = \
            np.arange(0.,Nw,1.)*dw + wmin
            
        cdef int nvis = u.shape[0]
    
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] Vis = \
            np.zeros(nvis, dtype=DTYPE)
        
        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)
        # Grid in u and v
        cdef double Du = W*du
        cdef double Dv = W*dv
        cdef double Dw = W*dw
        
        cdef Py_ssize_t i, j, urang, vrang, wrang, k, Wu, Wv, l
        
        cdef double gcf_val_u, gcf_val_v, gcf_val_w, gcf_val, upos, vpos
        cdef double delu, delv, delw, temp


        wrang = int(np.ceil((w - 0.5*Dw - wmin)/dw))
    
        for k in range(nvis):
            upos = u[k]
            vpos = v[k]
            
            urang = int(np.ceil((upos - 0.5*Du - umin)/du))
            vrang = int(np.ceil((vpos - 0.5*Dv - vmin)/dv))
            
    
            for i in range(urang, urang+W):
                if i>=Nu or i<0: continue
                delu = upos - ugrid[i]

                gcf_val_u = gcf_kaiser(delu, Du, beta)
                
                for j in range(vrang, vrang+W):
                    if j>=Nv or j<0: continue
                    delv = vpos-vgrid[j]

                    gcf_val_v = gcf_kaiser(delv, Dv, beta)
                    
                    for l in range(wrang, wrang+W):
                        if l>=Nw or l<0: continue
                    
                        delw = w-wgrid[l]

                        gcf_val_w = gcf_kaiser(delw, Dw, beta)
                        
                        gcf_val = gcf_val_u*gcf_val_v*gcf_val_w                                                
                        Vis[k] = Vis[k] + regVis[l,j,i]*gcf_val
    
        return Vis
        
        
        
        
        
def degrid_3d_pol(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v, \
    double w, np.ndarray[CTYPE_t, ndim=3] regVis, \
    double du, double Nu, double umin, double dv, double Nv, double vmin, \
    double dw, double Nw, double wmin, double alpha, int W):

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
            np.arange(0.,Nu,1.)*du + umin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vgrid = \
            np.arange(0.,Nv,1.)*dv + vmin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] wgrid = \
            np.arange(0.,Nw,1.)*dw + wmin
            
        cdef int nvis = u.shape[0]
    
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] VisQ = \
            np.zeros(nvis, dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] VisU = \
            np.zeros(nvis, dtype=CTYPE)
        
        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)
        # Grid in u and v
        cdef double Du = W*du
        cdef double Dv = W*dv
        cdef double Dw = W*dw
        cdef double gcf_val_u, gcf_val_v, gcf_val_w
        
        cdef Py_ssize_t i, j, urang, vrang, wrang, k, Wu, Wv, l
        
        cdef double gcf_val
        
        cdef complex pos_visval, neg_visval
        cdef complex visval_refresh = complex(0,0)

        wrang = int(np.ceil((w - 0.5*Dw - wmin)/dw))
    
        for k in range(nvis):
            
            pos_visval = visval_refresh
            neg_visval = visval_refresh
    
            urang = int(np.ceil((u[k] - 0.5*Du - umin)/du))
            vrang = int(np.ceil((v[k] - 0.5*Dv - vmin)/dv))
    
            for i in range(urang, urang+W):
                if (i>=Nu or i<0): continue
                gcf_val_u = gcf_kaiser(u[k]-ugrid[i], Du, beta)
                
                for j in range(vrang, vrang+W):
                    if (j>=Nv or j<0): continue
                    gcf_val_v = gcf_kaiser(v[k]-vgrid[j], Dv, beta)
                    
                    for l in range(wrang, wrang+W):
                         if (l>=Nw or l<0): continue
                         gcf_val_w = gcf_kaiser(w-wgrid[l], Dw, beta)
                         
                         gcf_val = gcf_val_u*gcf_val_v*gcf_val_w
                         pos_visval = pos_visval + regVis[l,j,i]*gcf_val
            
            urang = int(np.ceil((-1*u[k] - 0.5*Du - umin)/du))
            vrang = int(np.ceil((-1*v[k] - 0.5*Dv - vmin)/dv))
                                
            for i in range(urang, urang+W):
                if (i>=Nu or i<0): continue
                gcf_val_u = gcf_kaiser(-1*u[k]-ugrid[i], Du, beta)
                
                for j in range(vrang, vrang+W):
                    if (j>=Nv or j<0): continue
                    gcf_val_v = gcf_kaiser(-1*v[k]-vgrid[j], Dv, beta)                    
                    
                    for l in range(wrang, wrang+W):
                         if (l>=Nw or l<0): continue
                         gcf_val_w = gcf_kaiser(w-wgrid[l], Dw, beta)
                         
                         gcf_val = gcf_val_u*gcf_val_v*gcf_val_w
                         neg_visval = neg_visval + regVis[l,j,i]*gcf_val
                                                
            
            
            
            VisQ[k] = VisQ[k] + complex(0.5*(pos_visval.real + neg_visval.real), \
                0.5*(pos_visval.imag - neg_visval.imag))
            
            VisU[k] = VisU[k] + complex(0.5*(neg_visval.imag + pos_visval.imag), \
                0.5*(-1.*pos_visval.real + neg_visval.real))
    
        return VisQ, VisU
        
        
        
        
        
        
        


def get_grid_corr(np.ndarray[DTYPE_t,ndim=1] x, np.ndarray[DTYPE_t,ndim=1] y,\
    np.ndarray[DTYPE_t,ndim=1] z, double du, double dv, double dl2, int W, \
    double alpha):
    
    cdef int Nz = z.shape[0]
    cdef int Ny = y.shape[0]
    cdef int Nx = x.shape[0]
    
    cdef np.ndarray[DTYPE_t,ndim=3, mode='c'] gridcorr = np.zeros([Nz, Ny, Nx],\
        dtype=DTYPE)
        
    # see Beatty et al. (2005)
    cdef double beta = np.pi*np.sqrt((W/alpha)**2.*(alpha - 0.5)**2 - 0.8) 
    
    cdef Py_ssize_t i, j, k
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gridcorr[k,j,i] = inv_gcf_kaiser(x[i], du, W, beta)*\
                    inv_gcf_kaiser(y[j], dv, W, beta)*\
                    inv_gcf_kaiser(z[k], dl2, W, beta)
                    
    return gridcorr
    

def get_grid_corr_slice(np.ndarray[DTYPE_t,ndim=1] x, \
    np.ndarray[DTYPE_t,ndim=1] y, \
    double z, double du, double dv, double dl2, int W, \
    double alpha):
    
#    cdef int Nz = z.shape[0]
    cdef int Ny = y.shape[0]
    cdef int Nx = x.shape[0]
    
    cdef np.ndarray[DTYPE_t,ndim=2, mode='c'] gridcorr = np.zeros([Ny, Nx],\
        dtype=DTYPE)
        
    # see Beatty et al. (2005)
    cdef double beta = np.pi*np.sqrt((W/alpha)**2.*(alpha - 0.5)**2 - 0.8) 
    
    cdef Py_ssize_t i, j
    
    for i in range(Nx):
        for j in range(Ny):
            gridcorr[j,i] = inv_gcf_kaiser(x[i], du, W, beta)*\
                inv_gcf_kaiser(y[j], dv, W, beta)*\
                inv_gcf_kaiser(z, dl2, W, beta)
                    
    return gridcorr
    
    
    
def sample_grid_pol(np.ndarray[DTYPE_t,ndim=1] ug, \
    np.ndarray[DTYPE_t,ndim=1] vg, np.ndarray[DTYPE_t,ndim=1] l2g, \
    np.ndarray[CTYPE_t,ndim=1] Qvisg, np.ndarray[CTYPE_t,ndim=1] Uvisg, \
    np.ndarray[CTYPE_t,ndim=3] gv, \
    double umin, double vmin, double l2min, double du, double dv, double dl2):

    
        cdef Py_ssize_t i, undx, vndx, l2ndx
        cdef int N = ug.shape[0]
        cdef double temp = 0

        cdef int Nu = gv.shape[2]
        cdef int Nv = gv.shape[1]
        cdef int Nl2 = gv.shape[0]
            
        for i in range(N):
            # compute the location for the visibility in the visibility cube
            temp = (ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (vg[i] - vmin)/dv + 0.5
            vndx = int(temp)
            temp = (l2g[i] - l2min)/dl2 + 0.5
            l2ndx = int(temp)
            
            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv)\
                and (l2ndx >= 0 and l2ndx < Nl2):
                    gv[l2ndx, vndx, undx].real = gv[l2ndx, vndx, undx].real + \
                        (Qvisg[i].real - Uvisg[i].imag)
                    gv[l2ndx, vndx, undx].imag = gv[l2ndx, vndx, undx].imag + \
                        (Qvisg[i].imag + Uvisg[i].real)
                    
                    
            # now compute the location for the -u,-v,l2 visibility, which is
            # equal to the complex conj of the u,v,l2 visibility if we 
            # assume that the individual Stokes images in Faraday space are real
            temp = (-1.*ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (-1.*vg[i] - vmin)/dv + 0.5
            vndx = int(temp)
            
            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv)\
                and (l2ndx >= 0 and l2ndx < Nl2):
                    gv[l2ndx, vndx, undx].real = gv[l2ndx, vndx, undx].real + \
                        (Qvisg[i].real + Uvisg[i].imag)
                    gv[l2ndx, vndx, undx].imag = gv[l2ndx, vndx, undx].imag + \
                        (Uvisg[i].real - Qvisg[i].imag)




def sample_grid(np.ndarray[DTYPE_t,ndim=1] ug, \
    np.ndarray[DTYPE_t,ndim=1] vg, np.ndarray[DTYPE_t,ndim=1] l2g, \
    np.ndarray[DTYPE_t,ndim=1] visg, \
    np.ndarray[DTYPE_t,ndim=3] gv, \
    double umin, double vmin, double l2min, double du, double dv, double dl2):

    
        cdef Py_ssize_t i, undx, vndx, l2ndx
        cdef int N = ug.shape[0]
        cdef double temp = 0

        cdef int Nu = gv.shape[2]
        cdef int Nv = gv.shape[1]
        cdef int Nl2 = gv.shape[0]
                 
            
        for i in range(N):
            # compute the location for the visibility in the visibility cube
            temp = (ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (vg[i] - vmin)/dv + 0.5
            vndx = int(temp)
            temp = (l2g[i] - l2min)/dl2 + 0.5
            l2ndx = int(temp)
            
            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv)\
                and (l2ndx >= 0 and l2ndx < Nl2):
                    gv[l2ndx, vndx, undx] = gv[l2ndx, vndx, undx] + visg[i]
                    
                    
            # now compute the location for the -u,-v,-l2 visibility, which is
            # equal to the complex conj of the u,v,l2 visibility if we 
            # assume that the individual Stokes images in Faraday space are real
            temp = (-1.*ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (-1.*vg[i] - vmin)/dv + 0.5
            vndx = int(temp)
            
            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv)\
                and (l2ndx >= 0 and l2ndx < Nl2):
                    gv[l2ndx, vndx, undx] = gv[l2ndx, vndx, undx] + \
                        visg[i].conjugate()






#### Internal functions below #######

cdef inline void grid_1d_complex_scalar(double x, CTYPE_t Qvis,\
    CTYPE_t Uvis, double dx, int W, double beta, double y, double z,\
    np.ndarray[DTYPE_t,ndim=1] x2, \
    np.ndarray[CTYPE_t,ndim=1] Qvis2, np.ndarray[CTYPE_t,ndim=1] Uvis2,\
    np.ndarray[DTYPE_t,ndim=1] y2, np.ndarray[DTYPE_t,ndim=1] z2):

   
        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = 1 
                

        cdef double Dx = W*dx

        cdef Py_ssize_t xndx
        
        cdef double xref, xg, gcf_val

            
        xref = ceil((x - 0.5*W*dx)/dx)*dx


        for xndx in range(W):
           
            xg = xref + xndx*dx


            
            gcf_val = gcf_kaiser(xg-x, Dx, beta)

            Qvis2[xndx] = Qvis*gcf_val
            Uvis2[xndx] = Uvis*gcf_val
            x2[xndx] = xg
            y2[xndx] = y
            z2[xndx] = z
            



cdef inline void grid_1d_complex(np.ndarray[DTYPE_t,ndim=1] x, \
    np.ndarray[CTYPE_t,ndim=1] Qvis, np.ndarray[CTYPE_t,ndim=1] Uvis, \
    double dx, int W, double beta, \
    np.ndarray[DTYPE_t,ndim=1] y, np.ndarray[DTYPE_t,ndim=1] z, \
    np.ndarray[DTYPE_t,ndim=1] x2,\
    np.ndarray[CTYPE_t,ndim=1] Qvis2, np.ndarray[CTYPE_t,ndim=1] Uvis2,\
    np.ndarray[DTYPE_t,ndim=1] y2, np.ndarray[DTYPE_t,ndim=1] z2): 
    
        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = x.shape[0]

        cdef double Dx = W*dx
        
        cdef Py_ssize_t indx, xndx, kndx
        
        cdef double xval, yval, zval, xref, xg, gcf_val
        
        cdef CTYPE_t Qvisval, Uvisval
        
        for indx in range(N):
            Qvisval = Qvis[indx]
            Uvisval = Uvis[indx]
            
            xval = x[indx]
            yval = y[indx]
            zval = z[indx]
            
            xref = ceil((xval - 0.5*W*dx)/dx)*dx
            for xndx in range(W):
               
                xg = xref + xndx*dx
                
                kndx = indx*W + xndx
                
                gcf_val = gcf_kaiser(xg-xval, Dx, beta)
                
                Qvis2[kndx] = Qvisval*gcf_val
                Uvis2[kndx] = Uvisval*gcf_val
                x2[kndx] = xg
                y2[kndx] = yval
                z2[kndx] = zval
                
                
                









cdef inline void grid_1d_scalar(double x, DTYPE_t vis,\
    double dx, int W, double beta, double y, double z,\
    np.ndarray[DTYPE_t,ndim=1] x2, \
    np.ndarray[DTYPE_t,ndim=1] vis2, \
    np.ndarray[DTYPE_t,ndim=1] y2, np.ndarray[DTYPE_t,ndim=1] z2):

   
        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = 1 
                

        cdef double Dx = W*dx

        cdef Py_ssize_t xndx
        
        cdef double xref, xg, gcf_val

            
        xref = ceil((x - 0.5*W*dx)/dx)*dx


        for xndx in range(W):
           
            xg = xref + xndx*dx


            
            gcf_val = gcf_kaiser(xg-x, Dx, beta)

            vis2[xndx] = vis*gcf_val
            x2[xndx] = xg
            y2[xndx] = y
            z2[xndx] = z
            



cdef inline void grid_1d(np.ndarray[DTYPE_t,ndim=1] x, \
    np.ndarray[DTYPE_t,ndim=1] vis, \
    double dx, int W, double beta, \
    np.ndarray[DTYPE_t,ndim=1] y, np.ndarray[DTYPE_t,ndim=1] z, \
    np.ndarray[DTYPE_t,ndim=1] x2,\
    np.ndarray[DTYPE_t,ndim=1] vis2, \
    np.ndarray[DTYPE_t,ndim=1] y2, np.ndarray[DTYPE_t,ndim=1] z2): 
    
        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = x.shape[0]

        cdef double Dx = W*dx
        
        cdef Py_ssize_t indx, xndx, kndx
        
        cdef double xval, yval, zval, xref, xg, gcf_val
        
        cdef DTYPE_t visval
        
        for indx in range(N):
            visval = vis[indx]
            
            xval = x[indx]
            yval = y[indx]
            zval = z[indx]
            
            xref = ceil((xval - 0.5*W*dx)/dx)*dx
            for xndx in range(W):
               
                xg = xref + xndx*dx
                
                kndx = indx*W + xndx
                
                gcf_val = gcf_kaiser(xg-xval, Dx, beta)
                
                vis2[kndx] = visval*gcf_val
                x2[kndx] = xg
                y2[kndx] = yval
                z2[kndx] = zval





cdef inline double get_beta(int W, double alpha):
    cdef double pi = 3.141592653589793
    # see Beatty et al. (2005)
    cdef double beta = pi*sqrt((W*W/alpha/alpha)*(alpha - 0.5)*(alpha - 0.5) - 0.8)
    
    return beta


cdef inline double gcf_kaiser(double k, double Dk, double beta):
    
    cdef double temp3 = 2.*k/Dk
    
    if (1 - temp3)*(1 + temp3) < -1e-12:
#        print "There is an issue with the gridding code!"
        raise Exception("There is an issue with the gridding code!")
    
    temp3 = sqrt(abs((1 - temp3)*(1 + temp3)))
    
    temp3 = beta*temp3
    
    cdef double C = (1./Dk)*gsl_sf_bessel_I0(temp3)
    
    return C


cdef inline double inv_gcf_kaiser(double x, double dk, int W, double beta):
    
    cdef double pi = 3.141592653589793
    cdef double temp1 = (pi*W*dk*x)**2.
    cdef double temp2 = beta**2.
    cdef double temp, c
    
    temp = sqrt(temp2 - temp1)
    c = (exp(temp) - exp(-1.*temp))/2./temp
    
    if temp1>temp2:
        temp = sqrt(temp1 - temp2)
        c = -0.5*(exp(-1.*temp) - exp(temp))/temp
#        print "There is trouble"

    return c

