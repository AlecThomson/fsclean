"""
FSCleanPM.py

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

An class to read, parse, and manage parset files for fsclean. This class
extends ParsetManager and defines the parameters that are required for
fsclean. It also has specific methods for managing the fsclean parameters,
like a method to return axis vectors.

General parset syntax:

    One parameter per line
    Seperated by the delimiter character (defined in the base class)
    Empty lines are OK
    Commented lines must begin with the comment character
"""

# leave here while testing
#import sys
#sys.path.append('/home/mrbell/Work/code/')

from pyrat.ParsetManager import ParsetManager

VERSION = "0.1.0.0"


class FSCleanPM(ParsetManager):
    """
    """

    parset_def = {"cellsize": (float, None, "The R.A. and Dec. pixel sizes " +
                      "(arcsec). [None]."),
                  "dphi": (float, None, "The Faraday depth pixel size " +
                      "(rad/m/m). [None]"),
                  "nra": (int, None, "Number of R.A. pixels. [None]"),
                  "ndec": (int, None, "Number of Dec. pixels. [None]"),
                  "nphi": (int, None, "Number of Faraday depth pixels. " +
                      "[None]"),
                  "niter": (int, 500, "Number of CLEAN iterations. [500]"),
                  "gain": (float, 0.1, "Gain parameter for CLEAN. [0.1]"),
                  "cutoff": (float, 0., "Cutoff for CLEAN. (Jy/beam) [0.]"),
                  "bmaj": (float, 0., "Restoring beam major axis FWHM on " +
                      "the sky plane. Use 0 to fit to dirty beam. " +
                      "(arcsec) [0.]"),
                  "bmin": (float, 0., "Restoring beam minor axis FWHM on " +
                      "the sky plane. Use 0 to fit to dirty beam. " +
                      "(arcsec) [0.]"),
                  "bpa": (float, 0., "Restoring beam position angle on " +
                      "the sky plane. Positive rotation defined north " +
                      "through east. (degrees) [0.]"),
                  "bphi": (float, 0., "Restoring beam FWHM in Faraday " +
                      "depth. Use 0 to fit to dirty beam. " +
                      "(rad/m/m) [0.]"),
                  "verbosity": (int, 1, "Verbosity level for output. [1]"),
                  "read_buffer": (int, 1000, "Number of rows to read from " +
                      "the MS file at a time. [1000]"),
                  "scratch_dir": (str, "./", "The dirctory in which to " +
                      "store scratch files. [./]"),
                  "clear_scratch": (int, 1, "Delete all scratch files on" +
                      " exit? [1]"),
                  "ms_column": (str, "DATA", "The MeasurementSet data " +
                      "column from which to read the visibility " +
                      "data. [DATA]"),
                  "grid_w": (int, 6, "Width, in number of pixels, of the " +
                      "grid convolution function. [6]"),
                  "grid_alpha": (float, 1.5, "Oversampling ratio for the " +
                      "gridding code. [1.5]"),
                  "clean_type": (int, 0, "Type of CLEAN algorithm to use. " +
                      "0 = Clark, 1 = Hogbom. [0]"),
                  "beam_patch_frac": (int, 4, "CLEAN will use a patch of " +
                      "the beam that is 1/beam_patch_frac " +
                      "times each image dimension. [4]")}
