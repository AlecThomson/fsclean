#!/usr/bin/env python

"""
fsclean.py

Faraday synthesis using 3D CLEAN deconvolution

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

Software for imaging the Faraday spectrum, i.e. the 3D distribution of
polarized intensity as a function of Faraday depth and position on the sky.
Imaging is performed using the Faraday synthesis technique (see Bell and
Ensslin (2012) for details) and therefore inherently in 3D. Deconvolution is
carried out using a 3D CLEAN algorithm.

Data is read from MeasurementSet files of the type used by CASA. Images are
written to FITS files.
"""

# leave here while testing
import sys
sys.path.append('/home/mrbell/Work/code/')

import os
import datetime
import numpy as np
from optparse import OptionParser

from FSCData import FSCData, FSCPolData
from FSCImage import FSCImage
from FSCleanPM import FSCleanPM

import pyrat.Messenger as M
from pyrat.RAImage import GridParams
from pyrat.RAData import read_data_from_ms
from pyrat.Constants import *

VERSION = '0.1.0.0'


class FSCoords(object):
    """
    """

    def __init__(self, pm):
        """
        Takes a parset manager class instance and computes all coordinate
        values required.

        Args:

        Returns:
        """

        # Requested image plane grid parameters
        dphi = pm.parset['dphi']
        dra = pm.parset['cellsize'] * ARCSEC_TO_RAD
        ddec = pm.parset['cellsize'] * ARCSEC_TO_RAD
        nphi = pm.parset['nphi']
        nra = pm.parset['nra']
        ndec = pm.parset['ndec']

        self.grid_def = [(dphi, nphi), (ddec, ndec), (dra, nra)]

        # Gridding parameters
        self.grid_params = GridParams(pm.parset['grid_alpha'],
                                      pm.parset['grid_w'])


class FSClean(object):
    """
    """
    # CLEAN algorithm types
    CLARK = 0
    HOGBOM = 1

    def __init__(self, pm=None):
        """
        Initialize the FSClean imager. Sets common values and inits the
        messenger class.

        Args:
            parset: FSCleanPM class instance, with the parset dict already
                loaded

        Returns:
            Nothing
        """
        if pm is None:
            self.m = M.Messenger()
            return

        self.pm = pm

        # Internal verbosity level convention
        # -1 off
        #  0 Warnings, Errors, Headers, Basic information
        #  1 Useful diagnostic information for most users
        #  2 Detailed diagnostic information for users
        #  3 Developer diagnostics
        #  4 Temporary print statements

        self.m = M.Messenger(self.pm.parset['verbosity'], use_color=True,
                             use_structure=False, add_timestamp=True)

        self.coords = FSCoords(pm)
        self.K = 1.     # Normalization constant for data to image transform
        self.Kinv = 1.  # Normalization constant for image to data transform
        self._scratch_files = []
        self.do_clean = False
        if self.pm.parset['niter'] > 0:
            self.do_clean = True

    def condense_cc_list(self, cc):
        """
        Desc.

        Args:

        Returns:
        """
        tcc = list(cc)
        cc_redux = []

        while len(tcc) > 0:
            temp = tcc.pop()
            topop = []
            for i in range(len(tcc)):
                if temp[0] == tcc[i][0] and temp[1] == tcc[i][1] \
                    and temp[2] == tcc[i][2]:
                        temp2 = tcc[i]
                        topop.append(i)
                        temp[3] += temp2[3]

            cc_redux.append(temp)

            topop.sort(reverse=True)
            for i in range(len(topop)):
                tcc.pop(topop[i])

        return cc_redux

    def run(self, msfn, outfn_base):
        """
        The main routine.

        Args:
            msfn: MeasurementSet file name
            outfn_base: base name for the output files

        Returns:

        """

        clean_funcs = {self.CLARK: self.clark_clean,
                       self.HOGBOM: self.hogbom_clean}

        self.ofnbase = outfn_base
        self.sfnbase = os.path.join(self.pm.parset['scratch_dir'],
                                    os.path.basename(outfn_base))

        imfn = self.ofnbase + '_im.hdf5'
        dbfn = self.ofnbase + '_db.hdf5'

        self.m.header1("Starting FSCLEAN v." + VERSION)
        self.m.message("Requested parameters:", 0)
        if self.m.verbosity >= 0:
            self.pm.print_parset()

        self.m.message("Initializing data objects...", 1)

        weights = FSCData(self.sfnbase + '_weights.hdf5',
                          np.dtype('float64'),
                          m=self.m)
        self.register_scratch_files([weights.fn, weights.coords.fn])

        vis = FSCPolData(self.sfnbase + '_vis.hdf5',
                         coords=weights.coords,
                         m=self.m)
        self.register_scratch_files([vis.Q.fn, vis.U.fn])

        im = FSCImage(imfn, np.dtype('complex128'),
                      self.coords.grid_def, self.coords.grid_params, m=self.m)
        self.register_scratch_files([im.osim.fn, im.fourier_grid.fn])

        db = FSCImage(dbfn, np.dtype('complex128'),
                      self.coords.grid_def, self.coords.grid_params, m=self.m,
                      grid_dtype=np.dtype('float64'))
        self.register_scratch_files([db.osim.fn, db.fourier_grid.fn])

        read_data_from_ms(msfn, vis, weights, self.pm.parset['ms_column'],
                          'WEIGHT', mode='pol')

        self.m.message("Setting l2min", 3)
        l2min = vis.coords.get_min_freq()
        l2min = l2min - \
            self.coords.grid_params.W * 2. * im.fourier_grid.deltas[0]
#        if l2min < 0.:
#            l2min = 0.

        im.fourier_grid.set_mincoord(0, l2min)
        db.fourier_grid.set_mincoord(0, l2min)

        self.m.message("l2min set to " + str(l2min) + " m^2", 3)

        self.m.message("Setting normalization...", 2)
        self.set_normalizations(weights, db)
        self.m.message("K is " + str(self.K), 3)
        self.m.message("Kinv is " + str(self.Kinv), 3)

        # Hand off data to the appropriate CLEAN function
        #[cc, resim] = self.clark_clean(vis, weights, im, db)
        [cc, resim] = clean_funcs[self.pm.parset['clean_type']](vis, weights,
                                                                im, db)

        # Write images and CC list to disk
        self.m.message("Writing metadata to image files.", 2)
        self.write_image_metadata(im, msfn)
        self.write_image_metadata(db, msfn)
        if resim is not None:
            self.write_image_metadata(resim, msfn)

        self.write_cclist(self.ofnbase + "_cclist.txt", cc)

        self.clean_up()

    def register_scratch_files(self, fns):
        """
        Desc.

        Args:

        Returns:

        """

        if isinstance(fns, str):
            self._scratch_files.append(fns)
        elif np.iterable(fns):
            self._scratch_files += fns
        else:
            raise TypeError('Cannot add the requested data type to ' +
                            'the scratch files list.')

    def write_cclist(self, fn, cc):
        """
        Desc.

        Args:

        Returns:

        """

        if not np.iterable(cc):
            self.m.warn("No clean components to write.")
            return

        self.m.message("Writing CLEAN component list to file.", 2)

        f = open(fn, 'w')
        for i in range(len(cc)):
            c = cc[i]
            line = "%d %d %d %f %f\n" % (c[0], c[1], c[2],
                                         c[3].real, c[3].imag)
            f.write(line)
        f.close()

    def set_normalizations(self, weights, db):
        """
        Set normalizations for transform and inverse transforms. Resets the
        class attributes K and Kinv.

        Args:
            weights: An FSCData object containing the weights for each
                visibility.
            db: An FSCImage object that will be used to store the dirty beam.
        Returns:
            Nothing.

        """

        if self.do_clean and self.pm.parset['clean_type'] == self.CLARK:
            self.m.message("Computing Kinv", 3)
            temp = FSCData(self.sfnbase + '_tempdata.hdf5',
                           coords=weights.coords,
                           dtype=np.dtype('float64'), m=self.m,
                           template=weights)
            self.register_scratch_files(temp.fn)

            [nphi, ndec, nra] = db.im.shape
            db.multiplywith(0.)
            db.im[nphi / 2, ndec / 2, nra / 2] = complex(1., 0.)
            db.transform(temp)

            nchan = 0.
            val = 0.
            for i in temp.iterkeys():
                freqs = temp.coords.get_freqs(i)
                for j in range(len(freqs)):
                    nchan += 1
                    val += np.mean(abs(temp.get_records(i, j)))
            self.Kinv = 1. / (val / nchan)

        self.m.message("Computing K", 3)
        weights.transform(db)
        self.K = 1. / db.find_max(abs)

    def clean_up(self):
        """
        Deletes all temp files created during imaging.

        Args:

        Returns:

        """
        if self.pm.parset['clear_scratch'] != 0:
            self.m.header2("Removing scratch files...")

            for i in range(len(self._scratch_files)):
                os.remove(self._scratch_files[i])

    def write_image_metadata(self, im, msfn):
        """
        Writes important parameters to the header of the image.

        Args:
            im: FSCImage object pointing to the file to write metadata to.
            msfn: Filename of the MeasurementSet containing the visibility data
                that has been imaged.
        Returns:
            Nothing.

        """

        from pyrap import tables
        if tables.tableexists(os.path.join(msfn, 'SOURCE')):
            pt = tables.table(os.path.join(msfn, 'SOURCE'))
            crval = pt.getcol('DIRECTION')[0]
            source_name = pt.getcol('NAME')[0]
        else:
            crval = [0., 0.]
            source_name = ''

        im.f.attrs['origin'] = 'fsclean v. ' + VERSION
        im.f.attrs['date'] = str(datetime.date.today())
        im.f.attrs['source'] = source_name
        im.f.attrs['axis_desc'] = ['Faraday Depth', 'Dec.', 'RA']
        im.f.attrs['axis_units'] = ['rad/m/m', 'rad', 'rad']
        im.f.attrs['image_units'] = 'Jy/beam'
        im.f.attrs['crpix'] = [im.im.shape[0] / 2, im.im.shape[1] / 2,
                               im.im.shape[2] / 2]
        im.f.attrs['cdelt'] = [im.deltas[0], im.deltas[1], im.deltas[2]]
        im.f.attrs['crval'] = [0., crval[1], crval[0]]

    def hogbom_clean(self, vis, weights, im, db):
        """
        The 3D Hogbom CLEAN algorithm.

        Args:
            vis: An FSCPolData object containing the stokes Q and U visibility
                data to be cleaned.
            weights: An FSCData object containing the weights for each
                visibility.
            im: An FSCImage object in which to store the cleaned image.
            db: An FSCImage object in which to store the dirty beam image.
                Should have 2x the image volume of im.

        Returns:
            A list of clean components. Each list entry contains a tuple of
            model locations (phi, dec, ra) defined in pixels, and the model
            flux.

        """

        self.m.header2("Started the Hogbom CLEAN routine...")

        # contains the oversized dirty beam (8x larger than normal one by vol)
        self.m.message("Computing oversized dirty beam...", 1)
        grid_def = self.coords.grid_def
        big_grid_def = list()
        for i in range(3):
            big_grid_def.append((grid_def[i][0], grid_def[i][1] * 2))

        bigdb = FSCImage(self.sfnbase + '_bigdb.hdf5', np.dtype('complex128'),
               big_grid_def, self.coords.grid_params,
               m=self.m, grid_dtype=np.dtype('float64'))
        self.register_scratch_files([bigdb.fn, bigdb.osim.fn,
                                     bigdb.fourier_grid.fn])

        weights.transform(bigdb)
        Kbig = 1. / bigdb.find_max(abs)
        bigdb.multiplywith(Kbig)

        # Works fine
        self.m.message("Computing dirty image...", 1)
        vis.multiplywith(weights)  # vis now contains the weighted data!
        vis.transform(im)
        # im will contain the residual image going forward
        im.multiplywith(self.K)

        if not self.do_clean:
            return None, None

        # object for holding the model point source image
        pointim = FSCImage(self.sfnbase + '_pointim.hdf5',
                           np.dtype('complex128'),
                           self.coords.grid_def, self.coords.grid_params,
                           m=self.m)
        self.register_scratch_files([pointim.fn, pointim.osim.fn,
                                     pointim.fourier_grid.fn])

        [nphi, nm, nl] = im.im.shape

        cutoff = self.pm.parset['cutoff']
        niter = self.pm.parset['niter']
        gain = self.pm.parset['gain']

        # will contain the shifted beam image scaled by the residual peak value
        tdb = FSCImage(self.sfnbase + '_tdb.hdf5', np.dtype('complex128'),
               self.coords.grid_def, self.coords.grid_params,
               m=self.m, grid_dtype=np.dtype('float64'))
        self.register_scratch_files([tdb.fn, tdb.osim.fn, tdb.fourier_grid.fn])

        cclist = list()
        N = 0
        total_flux = complex(0, 0)

        while True:

            [pphi, pm, pl] = im.find_argmax(abs)
            pval = im.im[pphi, pm, pl]

            if abs(pval) < cutoff:
                self.m.success("Stopping! Cutoff has been reached.")
                break

            total_flux = total_flux + pval * gain

            N += 1

            self.m.message(".    Iteration " + str(N), 2)
            self.m.message(".    CLEAN Component info:", 2)
            self.m.message(".    .    value: " + str(pval * gain), 2)
            self.m.message(".    .    abs. value: " +
                           str(abs(pval * gain)), 2)
            self.m.message(".    .    phi: " + str(pphi), 2)
            self.m.message(".    .    m: " + str(pm), 2)
            self.m.message(".    .    l: " + str(pl), 2)
            self.m.message(".    .    total pol. flux: " +
                           str(abs(total_flux)), 2)

            cclist.append([pphi, pm, pl, pval * gain])

            bigdb.copy_patch_to(tdb, (pphi, pm, pl))
            tdb.multiplywith(gain * pval)
            im.subtractoff(tdb)

            if N >= niter:
                self.m.success("Stopping! Maximum iterations reached.")
                break

        self.m.message("Adding CLEAN model to image...", 1)

        cclist = self.condense_cc_list(cclist)  # OK
        self.make_cclist_image(cclist, pointim)  # OK

        self.m.message("Convolving with CLEAN beam...", 1)

        self.make_beam_image(tdb)  # OK

        pointim.convolve_with(tdb)  # OK

        resim = FSCImage(self.sfnbase + '_resim.hdf5',
                         np.dtype('complex128'),
                         self.coords.grid_def, self.coords.grid_params,
                         m=self.m)

        im.copy_to(resim)
        self.register_scratch_files([resim.osim.fn, resim.fourier_grid.fn])

        im.addto(pointim)

        return cclist, resim

    def clark_clean(self, vis, weights, im, db):
        """
        The 3D Clark CLEAN algorithm.

        Args:
            vis: An FSCPolData object containing the stokes Q and U visibility
                data to be cleaned.
            weights: An FSCData object containing the weights for each
                visibility.
            im: An FSCImage object in which to store the cleaned image.
            db: An FSCImage object in which to store the dirty beam image.

        Returns:
            A list of clean components. Each list entry contains a tuple of
            model locations (phi, dec, ra) defined in pixels, and the model
            flux.

        """

        self.m.header2("Started the Clark CLEAN routine...")

        # Works fine
        self.m.message("Computing dirty beam...", 1)
        weights.transform(db)
        db.multiplywith(self.K)

        # Works fine
        self.m.message("Computing dirty image...", 1)
        vis.multiplywith(weights)  # vis now contains the weighted data!

        if not self.do_clean:
            # im will contain the residual image going forward
            vis.transform(im)
            im.multiplywith(self.K)
            return None, None

        # object for holding the model point source image
        pointim = FSCImage(self.sfnbase + '_pointim.hdf5',
                           np.dtype('complex128'),
                           self.coords.grid_def, self.coords.grid_params,
                           m=self.m)
        self.register_scratch_files([pointim.fn, pointim.osim.fn,
                                     pointim.fourier_grid.fn])

        # object for holding the model visibilities
        modelvis = FSCPolData(self.sfnbase + '_modelvis.hdf5',
                              coords=weights.coords, m=self.m, template=vis)
        self.register_scratch_files([modelvis.Q.fn, modelvis.U.fn])

        [nphi, nm, nl] = im.im.shape

        PFRAC = self.pm.parset['beam_patch_frac']
        cutoff = self.pm.parset['cutoff']
        niter = self.pm.parset['niter']
        gain = self.pm.parset['gain']

        # number of pixels along each axis of the beam patch
        pnphi = nphi / PFRAC
        pnm = nm / PFRAC
        pnl = nl / PFRAC

        self.m.message("Extracting beam patch and computing highest " +
                       "external sidelobe...", 2)
        tdb = FSCImage(self.sfnbase + '_tdb.hdf5', np.dtype('complex128'),
                       self.coords.grid_def, self.coords.grid_params,
                       m=self.m, grid_dtype=np.dtype('float64'))
        self.register_scratch_files([tdb.fn, tdb.osim.fn, tdb.fourier_grid.fn])

        db.copy_to(tdb)

        patch = tdb.im[nphi / 2 - pnphi / 2:nphi / 2 + pnphi / 2,
                       nm / 2 - pnm / 2:nm / 2 + pnm / 2,
                       nl / 2 - pnl / 2:nl / 2 + pnl / 2]

        # get only the rest of the beam outside of the patch
        tdb.im[nphi / 2 - pnphi / 2:nphi / 2 + pnphi / 2,
               nm / 2 - pnm / 2:nm / 2 + pnm / 2,
               nl / 2 - pnl / 2:nl / 2 + pnl / 2] = np.zeros((pnphi, pnm, pnl),
                                                  dtype=np.dtype('complex128'))

        # find the largest sidelobe external to the patch
        extsl = tdb.find_max(abs)
        # for test dataset, extsl should be 0.112

        self.m.message("Largest sidelobe level outside beam patch: " +
                       str(extsl), 3)

        cclist = list()
        stop = False
        N = 1
        total_flux = complex(0, 0)

        while True:
            # Major cycle
            self.m.message("Begin Major Cycle", 1)

            # im will contain the residual image going forward
            vis.transform(im)
            im.multiplywith(self.K)

            [pphi, pm, pl] = im.find_argmax(abs)
            pval = im.im[pphi, pm, pl]

            self.m.message("Initial residual map peak: " + str(abs(pval)), 1)
            if abs(pval) < cutoff:
                self.m.success("Stopping! Cutoff has been reached.")
                stop = True
                break
            slim = extsl * abs(pval)
            F = 1. + 1. / N
            tcclist = list()

            if abs(pval) < slim * F:
                slim = abs(pval) / F

            self.m.message("Initial minor cycle stop level: " +
                           str(slim * F), 2)

            while abs(pval) >= slim * F:
            # Minor cycle

                total_flux = total_flux + pval * gain

                self.m.message(".    Starting minor cycle " + str(N), 2)
                self.m.message(".    CLEAN Component info:", 2)
                self.m.message(".    .    value: " + str(pval * gain), 2)
                self.m.message(".    .    abs. value: " +
                               str(abs(pval * gain)), 2)
                self.m.message(".    .    phi: " + str(pphi), 2)
                self.m.message(".    .    m: " + str(pm), 2)
                self.m.message(".    .    l: " + str(pl), 2)
                self.m.message(".    .    total pol. flux: " +
                               str(abs(total_flux)), 2)

                tcclist.append([pphi, pm, pl, pval * gain])

                # find phimin/max, lmin/max, mmin/max, accounting for map edges
                # crop the patch if necessary (because it runs off the edge)
                phimax = pphi + pnphi / 2
                phimin = pphi - pnphi / 2
                pc_phi_low = 0
                pc_phi_high = pnphi
                if phimin < 0:
                    phimin = 0
                    # lower index of the cropped patch
                    pc_phi_low = pnphi / 2 - pphi
                if phimax > nphi:
                    phimax = nphi
                    # upper index of the cropped patch
                    pc_phi_high = pnphi / 2 + (nphi - pphi)

                mmax = pm + pnm / 2
                mmin = pm - pnm / 2
                pc_m_low = 0
                pc_m_high = pnm
                if mmin < 0:
                    mmin = 0
                    # lower index of the cropped patch
                    pc_m_low = pnm / 2 - pm
                if mmax > nm:
                    mmax = nm
                    # upper index of the cropped patch
                    pc_m_high = pnm / 2 + (nm - pm)

                lmax = pl + pnl / 2
                lmin = pl - pnl / 2
                pc_l_low = 0
                pc_l_high = pnl
                if lmin < 0:
                    lmin = 0
                    # lower index of the cropped patch
                    pc_l_low = pnl / 2 - pl
                if lmax > nl:
                    lmax = nl
                    # upper index of the cropped patch
                    pc_l_high = pnl / 2 + (nl - pl)

                tpatch = patch[pc_phi_low:pc_phi_high,
                               pc_m_low:pc_m_high,
                               pc_l_low:pc_l_high].copy()

                im.im[phimin:phimax, mmin:mmax, lmin:lmax] = \
                    im.im[phimin:phimax, mmin:mmax, lmin:lmax] - \
                    gain * pval * tpatch

                [pphi, pm, pl] = im.find_argmax(abs)
                pval = im.im[pphi, pm, pl]
                N += 1
                F += 1. / N
                if abs(pval) < cutoff or N > niter:
                    # Is this true?  Does the peak value found during the minor
                    # cycle count for the stop condition?  The residual image
                    # here is kind of meaningless
                    self.m.success("Stopping! Cutoff or niter " +
                                   "has been reached.")
                    stop = True
                    break

            if not stop:
                self.m.message("Minor cycle stop condition reached.", 1)

            self.m.message("Inverting model to vis. space and " +
                           "subtracting...", 1)
            tcclist = self.condense_cc_list(tcclist)
            self.make_cclist_image(tcclist, pointim)

            pointim.transform(modelvis)

            modelvis.multiplywith(self.Kinv)
            modelvis.multiplywith(weights)

            vis.subtractoff(modelvis)
            self.m.message("Done.", 1)

            cclist = cclist + tcclist

            if stop:
                break
        self.m.message("Inverting CLEAN model...", 1)
        del patch

        cclist = self.condense_cc_list(cclist)  # OK
        self.make_cclist_image(cclist, pointim)  # OK

        self.m.message("Convolving with CLEAN beam...", 1)
        self.make_beam_image(tdb)  # OK

        pointim.convolve_with(tdb)  # OK

        # construct residual image
        vis.transform(im)
        im.multiplywith(self.K)

        resim = FSCImage(self.sfnbase + '_resim.hdf5',
                         np.dtype('complex128'),
                         self.coords.grid_def, self.coords.grid_params,
                         m=self.m)

        im.copy_to(resim)
        self.register_scratch_files([resim.osim.fn, resim.fourier_grid.fn])

        im.addto(pointim)

        return cclist, resim

    def make_beam_image(self, beamim):
        """
        Desc.

        Args:

        Returns:

        """

        beamim.multiplywith(0.)

        ln2 = 0.693147181

        bmaj = self.pm.parset['bmaj']
        bmin = self.pm.parset['bmin']
        bphi = self.pm.parset['bphi']

        invsigmal2 = 8 * ln2 * bmaj ** -2.
        invsigmam2 = 8 * ln2 * bmin ** -2.
        invsigmaphi2 = 8 * ln2 * bphi ** -2.

        # the size of the image over which to compute the gaussian
        # zero outside
        denom = self.pm.parset['beam_patch_frac']

        [nphi, nm, nl] = beamim.im.shape

        patch = np.zeros((nphi / denom, nm / denom, nl / denom),
                         dtype=beamim.im.dtype)

        phic = patch.shape[0] / 2
        mc = patch.shape[1] / 2
        lc = patch.shape[2] / 2

        philow = nphi / 2 - phic
        phihigh = nphi / 2 + phic
        mlow = nm / 2 - mc
        mhigh = nm / 2 + mc
        llow = nl / 2 - lc
        lhigh = nl / 2 + lc

        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                for k in range(patch.shape[2]):
                    patch[i, j, k] = np.exp(-0.5 * (invsigmaphi2 *
                                            (i - phic) ** 2 +
                                            invsigmam2 * (j - mc) ** 2 +
                                            invsigmal2 * (k - lc) ** 2))

        beamim.im[philow:phihigh, mlow:mhigh, llow:lhigh] = patch

    def make_cclist_image(self, cclist, im):
        """
        Desc.

        Args:

        Returns:
            Nothing.
        """
        im.multiplywith(0.)
        # list entries... [phi, m, l, val]
        for i in range(len(cclist)):
            [phi, m, l, val] = cclist[i]
            # there must be a better way...
            im.im[phi, m, l] = im.im[phi, m, l] + val

if __name__ == '__main__':
    """
    Handle all parsing here if started from the command line, then pass off to
    the main routine.
    """

    desc = "Software for reconstructing the Faraday spectrum, i.e. the 3D " + \
        "distribution of polarized intensity as a function of Faraday depth" +\
        " and position on the sky, from full-polarization, multi-frequency " +\
        "visibility data. Imaging is conducted using the Faraday " + \
        "synthesis technique (for details see Bell and Ensslin, 2012). " + \
        "Deconvolution is " + \
        "carried out using a 3D Clark CLEAN algorithm. " + \
        "Data is read from MeasurementSet files of the type used by CASA. " + \
        "Images are written to HDF5 image files. See the README for info " + \
        "about the image file format."

    parser = OptionParser(usage="%prog <parset file> <in file> <out file>",
                          description=desc, version="%prog " + VERSION)

    parser.add_option("-p", "--parset_desc", action="store_true",
                      help="show parameter set file description and exit",
                      default=False)

    (options, args) = parser.parse_args()

    pm = FSCleanPM()

    if options.parset_desc:
        pm.print_help()
    else:
        if len(args) != 3:
            parser.error("Incorrect number of arguments.")
        pm.parse_file(args[0])
        fsc = FSClean(pm)
        fsc.run(args[1], args[2])
