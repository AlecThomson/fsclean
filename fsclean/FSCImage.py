"""
FSCImage.py

Image class for use in the fsclean package.

*******************************************************************************

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

*******************************************************************************

"""

# leave here while testing
#import sys
#sys.path.append('/home/mrbell/Work/code/')

from pyrat.RAImage import Image as PI
#from pyrat import RAData
from fsclean import FSCData as FD
import grid_tools
import numpy as np


class FSCImage(PI):
    """
    """

    def __init__(self, fn, dtype, coords, grid_params=None, m=None,
                 grid_dtype=None):
        """
        Desc.

        Args:

        Returns:

        """
        super(FSCImage, self).__init__(fn, dtype, coords, grid_params, m,
            grid_dtype)

        # The l2 axis is not zero centered, should be changed to l2min later
        self.fourier_grid.set_mincoord(0, 0.)

    def copy_patch_to(self, other, loc):
        """
        Copies a patch of the image to a smaller image, with the center of the
        larger image shifted to the given location in the smaller image.

        Args:
            other: The other image to copy to. Must be smaller than the current
                image.
            loc: Tuple of pixel locations in the smaller image that correspond
                to the center of the current image.

        Returns:
            None
        """
        # TODO: This function is a bit basic at the moment. Put in checks, etc.
        shp = other.im.shape

        for i in range(shp[0]):
            cutout = self.im[shp[0] - loc[0] + i,
                             shp[1] - loc[1]:2 * shp[1] - loc[1],
                             shp[2] - loc[2]:2 * shp[2] - loc[2]]
            other.im[i] = cutout

        return None

    def convolve_with(self, other):
        """
        Desc.

        Args:

        Returns:

        """
        # zeropad the image to prepare for fourier inversion
        self.zeropad_image()
        other.zeropad_image()

        if self.shape != other.shape:
            raise Exception("Cannot convolve images of different sizes.")

        # Transform in 1D along each LOS, in place
        for i in range(self.osim.shape[1]):
            im_slice = self.osim.im[:, i, :]
            oim_slice = other.osim.im[:, i, :]

            for j in range(self.osim.shape[2]):
                im_slice[:, j] = np.fft.ifft(np.fft.ifftshift(im_slice[:, j]))
                oim_slice[:, j] = np.fft.ifft(
                                    np.fft.ifftshift(oim_slice[:, j]))

            self.osim.im[:, i, :] = im_slice
            other.osim.im[:, i, :] = oim_slice

        # 2D fourier inversion onto Fourier grid
        for i in range(self.osim.shape[0]):
            gv_slice = self.osim.im[i]
            ogv_slice = other.osim.im[i]

            result = np.fft.fftshift(
                np.fft.fft2(np.fft.fftshift(gv_slice)))

            oresult = np.fft.fftshift(
                np.fft.fft2(np.fft.fftshift(ogv_slice)))

            if self.fourier_grid.dtype != np.dtype('complex128'):
                self.fourier_grid.im[i, :] = result.real
            else:
                self.fourier_grid.im[i, :] = result

            if other.fourier_grid.dtype != np.dtype('complex128'):
                other.fourier_grid.im[i, :] = oresult.real
            else:
                other.fourier_grid.im[i, :] = oresult

        self.fourier_grid.multiplywith(other.fourier_grid)

        # 2D fourier inversion onto oversized image
        for i in range(self.fourier_grid.shape[0]):
            gv_slice = self.fourier_grid.im[i]

            result = np.fft.ifftshift(
                np.fft.ifft2(np.fft.ifftshift(gv_slice)))

            self.osim.im[i, :] = result

        for i in range(self.osim.shape[1]):
            im_slice = self.osim.im[:, i, :]

            for j in range(self.osim.shape[2]):
                im_slice[:, j] = np.fft.fftshift(np.fft.fft(im_slice[:, j]))

            self.osim.im[:, i, :] = im_slice

        # To fix the normalization of the convolution procedure
        # multiply with the number of pixels in phi
        self.osim.multiplywith(self.osim.shape[0])

        self.crop_osimage()

    def transform(self, data):
        """
        Transforms a Faraday spectral cube onto a PolData object by Fourier
        transformation and degridding. The transformed Data will not be
        properly normalized. Normalization must be done externally.

        Args:
            data: An FSCData object in which to store the transformed image.

        Returns:
            Nothing.

        """

        if not isinstance(data, FD.FSCData) \
            and not isinstance(data, FD.FSCPolData):
                raise Exception("Unsupported RAData object used as target" +
                    " for image transform!")

        # init the data vector
        data.multiplywith(0.)

        [dl2, dv, du] = self.fourier_grid.deltas
        [Nl2, Nv, Nu] = self.fourier_grid.im.shape

        alpha = self.alpha
        W = self.W

        [l2min, vmin, umin] = self.fourier_grid.mincoords

        ii = complex(0, 1)

        # divide by grid correction
        self.m.message("Pre-degridding grid correction...", 2)
        # Grid correct
        [phi, dec, ra] = self.get_axes()
        for i in range(self.im.shape[0]):
            im_slice = self.im[i]
            gc_slice = grid_tools.get_grid_corr_slice(ra, dec, phi[i],
                                                      du, dv, dl2, W, alpha)
            im_slice = im_slice / gc_slice
            self.im[i, :] = im_slice

        self.m.message("Inverting image into data space...", 2)

        # zeropad the image to prepare for fourier inversion
        # function copies data to the osim attribute
        self.zeropad_image()

        # 1D fourier inversion in place
        phi_os = self.osim.get_axis(0)
        phase_shift = np.exp(2. * np.pi * ii * l2min * phi_os)
        for i in range(self.osim.shape[1]):
            im_slice = self.osim.im[:, i, :]
            for j in range(self.osim.shape[2]):
                im_slice[:, j] = np.fft.ifft(np.fft.ifftshift(im_slice[:, j]
                                    * phase_shift))

            if self.osim.dtype != np.dtype('complex128'):
                self.osim.im[:, i, :] = im_slice.real
            else:
                self.osim.im[:, i, :] = im_slice

        # 2D fourier inversion onto the fourier grid
        for i in range(self.osim.shape[0]):
            gv_slice = self.osim.im[i]
            result = np.fft.fftshift(
                np.fft.fft2(np.fft.fftshift(gv_slice)))

            if self.fourier_grid.dtype != np.dtype('complex128'):
                self.fourier_grid.im[i, :] = result.real
            else:
                self.fourier_grid.im[i, :] = result

        # degrid onto data vector one channel at a time
        self.m.message("Degridding data...", 2)

        for i in data.keys():
            # get the l2 values stored within this spw
            l2vec = data.coords.get_freqs(i)
            nchan = len(l2vec)

            # Loop over each channel, grid each independently
            for j in range(nchan):
                l2 = l2vec[j]

                l2ndx = int((l2 - self.fourier_grid.mincoords[0] +
                    0.5 * self.fourier_grid.deltas[0]) /
                    self.fourier_grid.deltas[0])

                [uvec, vvec] = data.coords.get_coords(i, j)

                # get fg slice from which to sample
                slab_min = l2ndx - W
                slab_max = l2ndx + W

                if slab_min >= self.fourier_grid.shape[0] or slab_min < 0 \
                    or slab_max < 0 or slab_max >= self.fourier_grid.shape[0]:
                        raise Exception("Something went wrong when " +
                            "calculating the lambda^2 index.")

                fg_slab = self.fourier_grid.im[slab_min:slab_max]

                slab_l2min = l2min + slab_min * dl2

                # Degrid and store in the data vector
                if isinstance(data, FD.FSCPolData):
                    drec = grid_tools.degrid_3d_pol(uvec, vvec, l2, fg_slab,
                        du, Nu, umin, dv, Nv, vmin, dl2, fg_slab.shape[0],
                        slab_l2min, alpha, W)

                else:
                    if self.fourier_grid.dtype != np.dtype('complex128'):
                        drec = grid_tools.degrid_3d(uvec, vvec, l2, fg_slab,
                            du, Nu, umin, dv, Nv, vmin, dl2, len(fg_slab),
                            slab_l2min, alpha, W)
                    else:
                        drec = grid_tools.degrid_3d_complex(uvec, vvec, l2,
                            fg_slab, du, Nu, umin, dv, Nv, vmin, dl2,
                            len(fg_slab), slab_l2min, alpha, W)

                data.store_records(drec, i, j)

        self.m.message("Finished degridding.", 2)
