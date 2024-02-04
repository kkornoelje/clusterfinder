'''
For tutorials look into
scott:/sptlocal/analysis/resources/spt3g_software_tutorial/ilc/*.html
Original notebooks are under
scott:/sptlocal/user/sri/analysis/spt3g_software_trials/ilc_branch/*.ipynb

Currently supports only analytic ILC.
1. Standard minimum variance ILC.
2. Constrained ILC (one or more components can be nulled).
3. Partial ILC (arXiv: https://arxiv.org/abs/2102.05033; Bleem, Crawford et al. 2021).
'''
import matplotlib.pyplot as plt

import numpy as np, sys, os, scipy as sc, re
from spt3g.mapspectra import basicmaputils
from spt3g import core
from spt3g.simulations import cmb, foregrounds as fg
from spt3g.mapspectra import basicmaputils as utils
from spt3g.mapspectra import map_analysis
from spt3g.beams import beam_analysis as beam_mod

import healpy as H
import clusterfunctions as clu

def get_multipole_weights(inver,fsz,ft_signal,deproj='None'):
    ny, nx,nbands = inver.shape[0], inver.shape[1], inver.shape[-1]
    inver= inver.reshape(ny*nx,nbands,nbands)

    
    profile = ft_signal.reshape(ny*nx,nbands)
    profile = profile[:,:,np.newaxis]
    avec= (ft_signal*fsz).reshape(ny*nx,nbands)
    
    if type(deproj) is str:
        bvec = np.asarray([1]*nbands)
        bvec = (ft_signal*bvec).reshape(ny*nx,nbands)
    else:
        bvec = (ft_signal*deproj).reshape(ny*nx,nbands)

        
    G = np.stack((avec, bvec),axis=-1)
    nr = np.einsum('ijk,ikl->ijl', inver, G)
    min_var_weight = nr.reshape(ny,nx,nbands,2)[:,:,:,0]
    dr = np.einsum('ijk,ikl->ijl', G.transpose(0, 2, 1),nr )

    drinv = np.linalg.pinv(dr)

    weight = np.einsum('ijk,ikl->ijl', nr, drinv)

    
    unit_weights = weight[:,:,0].reshape(ny,nx,nbands)
    null_weights = weight[:,:,1].reshape(ny,nx,nbands)


    return min_var_weight,unit_weights,null_weights, nr, drinv

def radial_profile(image, binsize, maxbin, minbin=0.0, xy=None, return_errors=False):
    """
    Get the radial profile of an image (both real and fourier space)

    Parameters
    ----------
    image : array
        Image/array that must be radially averaged.
    binsize : float
        Size of radial bins.  In real space, this is
        radians/arcminutes/degrees/pixels.  In Fourier space, this is
        \Delta\ell.
    maxbin : float
        Maximum bin value for radial bins.
    minbin : float
        Minimum bin value for radial bins.
    xy : 2D array
        x and y grid points.  Default is None in which case the code will simply
        use pixels indices as grid points.
    return_errors : bool
        If True, return standard error.

    Returns
    -------
    bins : array
        Radial bin positions.
    vals : array
        Radially binned values.
    errors : array
        Standard error on the radially binned values if ``return_errors`` is
        True.
    """

    image = np.asarray(image)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy

    radius = np.hypot(x, y)
    radial_bins = np.arange(minbin, maxbin, binsize)

    hits = np.zeros(len(radial_bins), dtype=float)
    vals = np.zeros_like(hits)
    errors = np.zeros_like(hits)

    for ib, b in enumerate(radial_bins):
        imrad = image[(radius >= b) & (radius < b + binsize)]
        total = np.sum(imrad != 0.0)
        hits[ib] = total

        if total > 0:
            # mean value in each radial bin
            vals[ib] = np.sum(imrad) / total
            errors[ib] = np.std(imrad)

    bins = radial_bins + binsize / 2.0

    std_mean = np.sum(errors * hits) / np.sum(hits)
    errors = std_mean / hits ** 0.5

    if return_errors:
        return bins, vals, errors
    else:
        return bins, vals

def plot_psi(cf,weights,nbands='',labels='',plot_sum=True,title='Matched Filter Weights'):
    if type(nbands) == str:
        nbands = cf.nbands
        
    ell_min = 0
    ell_max = np.pi / (cf.resrad)
    delta_ell = 2 * np.pi / (np.max([cf.nx,cf.ny]) * cf.resrad)
    ell_bins = np.arange(ell_min, ell_max + delta_ell, delta_ell)
    ell_plot=(np.array(ell_bins[1:]) + np.array(ell_bins[:-1]))/2

    pypsi=np.reshape(np.split(weights,nbands,axis=2),(nbands, 3360, 3360)) 
    py = []
    for i in range(len(pypsi)):
        py.append(basicmaputils.av_ps((-1*pypsi[i]),
                                     .25*core.G3Units.arcmin,
                                     ell_bins,s=(cf.ny,cf.nx),real=False))
    plt.rcParams['figure.figsize'] = (5,5)
    plt.title(title,fontsize=15)
    colors = ['red','green','blue']
    full_sum = np.zeros_like(py[0])
    for i in range(nbands):
        if type(labels) !=str:
            plt.plot(ell_plot,py[i],label=labels[i])
        else:
            plt.plot(ell_plot,py[i])
        full_sum+=(py[i]*cf.fsz[i])
    
    
    if plot_sum:
        plt.plot(ell_plot,full_sum,color='black',label='sum')
    plt.legend(fontsize=13)
    plt.xlim(0,10_000)

    
    return ell_plot, py
    
    
    