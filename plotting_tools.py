"""
plotting_tools.py

Utilities and scripts for procuding plots of Faraday spectra.

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

# leave here while testing
import sys
sys.path.append('/home/mrbell/Work/code/')

import pylab as pl
#from matplotlib.widgets import Slider
import h5py
from pyrat import RAImage
from pyrat.Constants import *
import numpy as np

cm = pl.cm.jet


DATASET_STRING = RAImage.DATASET_STRING

# Code snippet taken from
# http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg03724.html

class IndexTracker:
   def __init__(self, ax, f, vmax):
       self.ax = ax

       self.f = f
       self.X = f[DATASET_STRING]
       self.slices,rows,cols = self.X.shape
       self.ind  = self.slices/2

       self.ax.set_title('Faraday Spectrum, Pol. Intensity \n $\phi$='+\
           str((self.ind-self.slices/2)\
           *self.f.attrs['cdelt'][0])+" rad/m$^2$")

       self.im = ax.imshow(abs(self.X[self.ind,:,:]), origin='lower', cmap=cm, \
           vmax=vmax)
       self.ax.figure.colorbar(self.im)
       xticks = self.ax.get_xticks()
#       print xticks
       labels = []
       for i in range(len(xticks)):
           labels += [5.*xticks[i]]
       self.ax.set_xticklabels(labels)
       self.update()

   def onpress(self, event):

       if event.key=='right':
           self.ind = np.clip(self.ind+1, 0, self.slices-1)
       elif event.key=='left':
           self.ind = np.clip(self.ind-1, 0, self.slices-1)
           
       self.update()

   def update(self):
       self.im.set_data(abs(self.X[self.ind,:,:]))
#       self.ax.set_ylabel('slice %s'%self.ind)
       self.ax.set_title('Faraday Spectrum, Pol. Intensity \n $\phi$='+\
           str((self.ind-self.slices/2)\
           *self.f.attrs['cdelt'][0])+" rad/m$^2$")
       self.im.axes.figure.canvas.draw()



def fsbrowser(fn, vmin=None, vmax=None):
    """
    desc.

    Args:
        
    Returns:
        Nothing
    
    """
    
    im_file = h5py.File(fn)

    fig = pl.figure()
    ax = fig.add_subplot(111)
    
    if vmin==None:
        vmin = 0.
    if vmax==None:
        vmax = find_max(im_file[DATASET_STRING], abs)
    
    tracker = IndexTracker(ax, im_file, vmax)
    
    fig.canvas.mpl_connect('key_press_event', tracker.onpress)
    
    pl.show()
    

def find_max(dset, func):
    vmax = None
    for i in range(dset.shape[0]):
        if vmax is None or np.max(func(dset[i].flatten())) > vmax:
            vmax = np.max(func(dset[i].flatten()))
    return vmax
    
def maxphiimage(fn, thresh):
    """
    A routine for making an 
    """
    pass
    
    
    
    
    
