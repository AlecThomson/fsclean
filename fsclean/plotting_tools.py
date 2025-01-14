"""
plotting_tools.py

Utilities and scripts for procuding plots of Faraday spectra.

******************************************************************************

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

******************************************************************************

"""

# leave here while testing
import sys
#sys.path.append('/home/mrbell/Work/code/')

import pylab as pl
#from matplotlib.widgets import Slider
import h5py
from pyrat import RAImage
from pyrat.Constants import *
import numpy as np
import subprocess as sp
from scipy.optimize import curve_fit

cm = pl.cm.jet

DATASET_STRING = RAImage.DATASET_STRING

# Code snippet for interactive plotting taken from
# www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg03724.html


class IndexTracker:
    """
    """

    def __init__(self, ax, f, vmax):
        """
        """
        self.ax = ax

        self.f = f
        self.X = f[DATASET_STRING]
        self.slices, rows, cols = self.X.shape
        self.ind = self.slices / 2

        self.ax.set_title('Faraday Spectrum, Pol. Intensity \n $\phi$=' +
            str((self.ind - self.slices / 2) *
            self.f.attrs['cdelt'][0]) + " rad/m$^2$")

        self.im = ax.imshow(abs(self.X[self.ind, :, :]), origin='lower',
                           cmap=cm, vmax=vmax)
        self.ax.figure.colorbar(self.im)
        self.update()

    def onpress(self, event):
        """
        """

        if event.key == 'right':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        elif event.key == 'left':
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1)

        self.update()

    def update(self):
        """
        """
        self.im.set_data(abs(self.X[self.ind, :, :]))
        self.ax.set_title('Faraday Spectrum, Pol. Intensity \n $\phi$=' +
                          str((self.ind - self.slices / 2) *
                          self.f.attrs['cdelt'][0]) + " rad/m$^2$")
        self.im.axes.figure.canvas.draw()


def fsbrowser(fn, vmin=None, vmax=None):
    """
    An interactive tool to move through a Faraday cube.

    Args:

    Returns:
        Nothing

    """

    im_file = h5py.File(fn)

    fig = pl.figure()
    ax = fig.add_subplot(111)

    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = find_max(im_file[DATASET_STRING], abs)

    tracker = IndexTracker(ax, im_file, vmax)

    fig.canvas.mpl_connect('key_press_event', tracker.onpress)

    pl.show()


def fsmovie(fn, mfn, vmin=None, vmax=None):
    """
    Make a video stepping through the Faraday depth axis of the cube.

    Args:
        fn: The file name of the source HDF5 file.
        mfn: The movie file name.
        vmin: The minimum of the color scale. [0]
        vmax: The maximum of the color scale. [max of the cube]

    Returns:
        None
    """

    im_file = h5py.File(fn)
    a = im_file[DATASET_STRING]

    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = find_max(a, abs)

    nframes = a.shape[0]

    for i in range(nframes):

        pl.figure()
        pl.imshow(abs(a[i, :, :]), vmax=vmax, vmin=vmin)
        pl.title('Faraday Spectrum, Pol. Intensity \n $\phi$=' +
            str((i - nframes / 2) *
            im_file.attrs['cdelt'][0]) + " rad/m$^2$")
        pl.colorbar()

        if i > 0:
            ndig = int(np.log10(i))
        else:
            ndig = 0
        istr = str(i)
        for j in range(3 - ndig):
            istr = "0" + istr

        fn_str = mfn + "_" + istr + ".png"
        pl.savefig(fn_str)
        pl.close()

        progress(20, i + 1, len(a))

    produce_avi(mfn, mfn + "_movie.avi")


def produce_avi(fnbase, outfile):
    """
    """
    results = sp.Popen(['ffmpeg', '-qscale', str(5), '-r', str(20), '-b',
                        str(9600), '-i', fnbase + '%04d.png', outfile],
                       stdout=sp.PIPE)
    out = results.communicate()
    if len(out[0]) != 0:
        print(out[0])


def find_max(dset, func):
    """
    """
    vmax = None
    for i in range(dset.shape[0]):
        slice_max = np.max(func(dset[i].flatten()))
        if vmax is None or slice_max > vmax:
            vmax = np.max(func(dset[i].flatten()))
    return vmax


def maxphiimage(fn, thresh):
    """
    A routine for making an "RM image" from a Faraday cube. Reports the Faraday
    depth of the max of the polarized emission along each LOS.

    Args:
        fn: File name of the source Faraday cube HDF5 file.
        thresh: Max brightness along the line of sight must be above this value
            in order to put a value in the map. Otherwise, the pixel will
            be blank.
    """

    # TODO: Add the ability to threshold as a mult. of noise stdv

    im_file = h5py.File(fn)
    a = im_file[DATASET_STRING]

    shape = (a.shape[1], a.shape[2])

    mask = np.zeros(shape)
    rmmap = np.zeros(shape)
    widmap = np.zeros(shape)

    myjet = pl.cm.jet
    myjet.set_bad('k')

    for i in range(shape[0]):
        imslice = a[:, i, :]
        for j in range(shape[1]):
            #maxval = max(abs(imslice[:, j]))
            #maxpix = np.argmax(abs(imslice[:, j]))

            los = imslice[:, j]
            try:
                maxpix, fwhm = find_peak(los, thresh)
            except RuntimeError:
                maxpix = np.argmax(abs(los))
                fwhm = -1

            if fwhm < 0:
                mask[i, j] = 1

            rmmap[i, j] = ((maxpix - a.shape[0] / 2) *
                           im_file.attrs['cdelt'][0])
            widmap[i, j] = fwhm * im_file.attrs['cdelt'][0]

        progress(20, i + 1, shape[0])

    maskedrm = np.ma.array(rmmap, mask=mask)
    maskedwid = np.ma.array(widmap, mask=mask)

    pl.figure()
    pl.imshow(maskedrm, cmap=myjet, origin='lower')
    pl.title("RM Map (values in rad/m$^2$)")
    pl.colorbar()
    pl.show()

    pl.figure()
    pl.imshow(maskedwid, cmap=myjet, origin='lower')
    pl.title("Peak FWHM (values in rad/m$^2$)")
    pl.colorbar()
    pl.show()

    return maskedrm, maskedwid


def progress(width, part, whole):
    """
    """

    percent = float(part) / float(whole)
    if percent > 1.:
        percent == 1.

    marks = int(width * (percent))
    spaces = width - marks
    loader = '[' + ('=' * marks) + (' ' * spaces) + ']'
    #sys.stdout.write("%s %d/%d %d%%\r" % (loader, part, whole, percent*100))
    sys.stdout.write("%s %d%%\r" % (loader, percent * 100.))
    if percent >= 1:
        sys.stdout.write("\n")
    sys.stdout.flush()


def find_peak(im, thresh):
    """
    Find max and argmax of an image with a single peak.

    Args:
        im: 1D Faraday spectrum image.

    Returns:
        xmax: location of maximum as a pixel number
        pmax: polarized intensity at maximum
        fwhm: fwhm of gaussian fit around maximum. Returns -1 if pmax is below
            the threshold.
    """

    halfwidth = 2

    loc = np.argmax(abs(im))
    val = max(abs(im))

    if val >= thresh:

        try:
            popt, pconv = curve_fit(gaussian,
                                np.arange(loc - halfwidth, loc + halfwidth),
                                abs(im[loc - halfwidth:loc + halfwidth]),
                                (val, loc, 4 * halfwidth))
        except ValueError:
            pl.figure()
            pl.plot(abs(im))
            pl.show()
            input("error found, hit enter to quit")
            raise

        xmax = popt[1]
        fwhms = popt[2] * 2.3548
    else:
        return loc, -1.

    return xmax, fwhms


def gaussian(x, a, b, c):
    """
    """

    val = a * np.exp(-(x - b) ** 2 / c ** 2)
    return val
