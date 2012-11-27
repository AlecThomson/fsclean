"""
FSCData.py

Data class for use in the fsclean package.

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

# TODO: FSCData and FSCPolData transform functions are almost exactly the same
#       Generalize and make sub-functions to reduce code duplication.
# TODO: Look this over to see whether the interface to the Data object needs to 
#       be modified. Perhaps there are functions that should be provided through
#       the interface that are being done by hand here.


# leave here while testing
import sys
sys.path.append('/home/mrbell/Work/code/')

#import RAImage as PI
from pyrat import RAData
import grid_tools
import numpy as np

class FSCData(RAData.Data):
    
    def transform(self, im):
        """
        Transforms visibility data onto a FSCImage object by gridding and Fourier 
        transformation. The transformed image will not be properly
        normalized. Normalization must be done externally.
        
        Args:
            image: An FSCImage object in which to store the transformed image.
            
        Returns:
            Nothing.
        
        """
        
        im.fourier_grid.init_with_scalar(0.)
        
        [dl2, dv, du] = im.fourier_grid.deltas
        
        alpha = im.alpha
        W = im.W
        
        [l2min, vmin, umin] = im.fourier_grid.mincoords
        
        ii = complex(0,1)
        
        self.m.message("Gridding data...",2)
        for i in self.iterkeys():
            # get the l2 values stored within this spw
            l2vec = self.coords.get_freqs(i)
            nchan = len(l2vec)
            
            # Loop over each channel, grid each independently
            for j in range(nchan):
                l2 = l2vec[j]
                l2ndx = int((l2 - l2min + 0.5*dl2)/dl2)
                
                [uvec, vvec] = self.coords.get_coords(i,j)
                dvec = self.get_records(i,j)
                
                # Convolve data vector with GCF
                [guvec, gvvec, gl2vec, gdvec] = grid_tools.grid_3d(\
                    uvec, vvec, l2, dvec, du, dv, dl2, alpha, W)
                
                # sample onto fg slice
                slab_min = l2ndx - W
                slab_max = l2ndx + W
                if slab_min  < 0:
                    slab_min = 0
                if slab_max > im.fourier_grid.shape[0]-1:
                    slab_max = im.fourier_grid.shape[0]-1
                
                if slab_min > im.fourier_grid.shape[0]-1 or slab_max < 0:
                    raise Exception("Something went wrong when calculating the"+\
                        " lambda^2 index.")
                
                fg_slab = im.fourier_grid.im[slab_min:slab_max]
                
                
                slab_l2min = l2min+slab_min*dl2
                
                grid_tools.sample_grid(guvec, gvvec, gl2vec, gdvec, \
                    fg_slab, umin, vmin, slab_l2min, du, dv, dl2)
                                    
                im.fourier_grid.im[slab_min:slab_max] = fg_slab
        
        self.m.message("Inverting Data...", 2)
        
        # Fourier transform onto osim
        for i in range(im.fourier_grid.shape[0]):
            # NOTE: Before and after an fft, do an fftshift to shift the zero 
            # frequency to the center of the spectrum 
            # Before and after an ifft, do an ifftshift
            gv_slice = im.fourier_grid.im[i]
            im.osim.im[i,:] = np.fft.ifftshift(\
                np.fft.ifft2(np.fft.ifftshift(gv_slice)))
            
        phi = im.osim.get_axis(0)
        phase_shift = np.exp(-2.*np.pi*ii*l2min*phi)
        for i in range(im.osim.shape[1]):
            im_slice = im.osim.im[:,i,:]
            for j in range(im.osim.shape[2]):
                im_slice[:,j] = np.fft.fftshift(np.fft.fft(im_slice[:,j]))\
                    *phase_shift
            im.osim.im[:,i,:] = im_slice
        
        self.m.message("Cropping image...", 2)
        # Clip
        im.crop_osimage()
        
        self.m.message("Grid correction...", 2)
        # Grid correct
        [phi, dec, ra] = im.get_axes()
        for i in range(im.im.shape[0]):
            im_slice = im.im[i]
            gc_slice = grid_tools.get_grid_corr_slice(ra, dec, phi[i], \
                du, dv, dl2, W, alpha)
            im_slice = im_slice/gc_slice
            im.im[i,:] = im_slice
        self.m.message("Finished grid correction.", 2)

     
class FSCPolData(RAData.PolData):
    
    def transform(self, im):
        """
        Transforms visibility data onto a FSCImage object by gridding and Fourier 
        transformation. The transformed image will not be properly
        normalized. Normalization must be done externally.
        
        Args:
            image: An FSCImage object in which to store the transformed image.
            
        Returns:
            Nothing.
        
        """
        
        im.fourier_grid.init_with_scalar(0.)
        
        [dl2, dv, du] = im.fourier_grid.deltas
        alpha = im.alpha
        W = im.W
        
        [l2min, vmin, umin] = im.fourier_grid.mincoords
        
        ii = complex(0,1)
        
        self.m.message("Gridding data...",2)
        for i in self.iterkeys():
            # get the l2 values stored within this spw
            l2vec = self.coords.get_freqs(i)
            nchan = len(l2vec)
            
            # Loop over each channel, grid each independently
            for j in range(nchan):
                l2 = l2vec[j]
                l2ndx = int((l2 - l2min + 0.5*dl2)/dl2)
                
                [uvec, vvec] = self.coords.get_coords(i,j)
                [qdvec, udvec] = self.get_records(i,j)
                
                # Convolve data vector with GCF
                [guvec, gvvec, gl2vec, gdqvec, gduvec] = grid_tools.grid_3d_pol(\
                    uvec, vvec, l2, qdvec, udvec, du, dv, dl2, alpha, W)
                
                # sample onto fg slice
                slab_min = l2ndx - W
                slab_max = l2ndx + W
                if slab_min  < 0:
                    slab_min = 0
                if slab_max > im.fourier_grid.shape[0]-1:
                    slab_max = im.fourier_grid.shape[0]-1
                
                if slab_min > im.fourier_grid.shape[0]-1 or slab_max < 0:
                    raise Exception("Something went wrong when calculating the"+\
                        " lambda^2 index.")
                        
                fg_slab = im.fourier_grid.im[slab_min:slab_max]
                
                
                slab_l2min = l2min+slab_min*dl2
                
                grid_tools.sample_grid_pol(guvec, gvvec, gl2vec, gdqvec, \
                    gduvec, fg_slab, umin, vmin, slab_l2min, du, dv, dl2)
                                    
                im.fourier_grid.im[slab_min:slab_max] = fg_slab
        
        self.m.message("Inverting Data...", 2)
        
        # Fourier transform onto osim
        for i in range(im.fourier_grid.shape[0]):
            # NOTE: Before and after an fft, do an fftshift to shift the zero 
            # frequency to the center of the spectrum 
            # Before and after an ifft, do an ifftshift
            gv_slice = im.fourier_grid.im[i]
            im.osim.im[i,:] = np.fft.ifftshift(\
                np.fft.ifft2(np.fft.ifftshift(gv_slice)))
            
        phi = im.osim.get_axis(0)
        phase_shift = np.exp(-2.*np.pi*ii*l2min*phi)
        for i in range(im.osim.shape[1]):
            im_slice = im.osim.im[:,i,:]
            for j in range(im.osim.shape[2]):
                im_slice[:,j] = np.fft.fftshift(np.fft.fft(im_slice[:,j]))\
                    *phase_shift
            im.osim.im[:,i,:] = im_slice
        
        self.m.message("Cropping image...", 2)
        # Clip
        im.crop_osimage()
        
        self.m.message("Grid correction...", 2)
        # Grid correct
        [phi, dec, ra] = im.get_axes()
        for i in range(im.im.shape[0]):
            im_slice = im.im[i]
            gc_slice = grid_tools.get_grid_corr_slice(ra, dec, phi[i], \
                du, dv, dl2, W, alpha)
            im_slice = im_slice/gc_slice
            im.im[i,:] = im_slice
        self.m.message("Finished grid correction.", 2)