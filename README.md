fsclean - Faraday Synthesis CLEAN imager
==========================================================

fsclean is a software package for producing 3D Faraday spectra using the Faraday
synthesis method, transforming directly from multi-frequency visibility data
to the Faraday depth-sky plane space. Deconvolution is accomplished using the
CLEAN algorithm.

Features include: 

  - Reads in MeasurementSet visibility data.
  - Clark style CLEAN algorithm. Inclusion of a Högbom style imager is planned.
  - Produces HDF5 formatted images. Simple matplotlib based visualization tools 
    are planned.
  - Handles images and data of arbitrary size, using scratch HDF5 files as 
    buffers for data that is not being immediately processed. Only limited by
    available HDD space.

For more information, see the Faraday synthesis paper [(Bell & Enßlin, 2012)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1112.4175).

fsclean is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl.html).
