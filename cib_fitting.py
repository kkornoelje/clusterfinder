import sys
import source_subtraction
sys.path.append("/home/kaylank/imports/")
from imports import *

from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy import constants as con

import numpy as np
import emcee
import matplotlib.pyplot as plt

def deconvolve_beam(input_map, beam,rebeam=False,psize=0.25):
    beam = copy.deepcopy(beam)
    beam[beam < 1e-2] = 1e50
    beam = 1/beam
    
    rebeam = copy.deepcopy(rebeam)
    
    deconvolved_map = clu.multi_band_filter([input_map], beam[:,:,np.newaxis], psize,1)
    if rebeam is False:
        return deconvolved_map
    else:
        reconvolved_map = clu.multi_band_filter([input_map], rebeam[:,:,np.newaxis], psize,1)
        return reconvolved_map

def mod_bb(freqs,T,amp,norm_freq = 150,beta=2):
    if T < 2.725 or T > 50:
        return np.inf
    if amp < 0:
        return np.inf
    norm_freq = norm_freq*u.GHz
    freqs*=u.GHz
    
    amp*=u.arcmin**2
    bb = BlackBody(temperature=T * u.K)
    flux = bb(freqs).to(u.mJy/u.arcmin**2)
    mod_bb_flux = (flux*amp) * (freqs/norm_freq)**beta
    return (mod_bb_flux).value


def calc_fsz_to_mJy(freqs,comp_y):
    if comp_y < 0:
        return np.inf
    else:
        f = np.asarray(freqs)*1e9*u.Hz
        T_cmb = 2.725*u.K
        x = ((con.h * f ) / (con.k_B * T_cmb)).value
        
        try:
            g_x = (x**4 *np.exp(x))/((np.exp(x) - 1)**2) *  clu.calc_fsz(f.value/1e9)
        except:
            g_x = 0
        
        I_0 = 2 * (con.k_B * T_cmb)**3 / ((con.h*con.c)**2)
        I_0/=u.s
        I_0 = I_0.to(u.W/u.m**2)
        I_0/=u.Hz
        I_0 = I_0.to(u.MJy)
        I_0 = I_0/u.sr
        I_0 = I_0.to(u.mJy/u.arcmin**2)


        return (I_0 * g_x * comp_y).value
    
def full_function(freqs,T,amp,comp_y):
    return mod_bb(freqs,T,amp) + calc_fsz_to_mJy(freqs,comp_y)


def log_likelihood(theta, x, y, yerr, func,param_bounds):
    if param_bounds is not None:
        for i in range(len(param_bounds)):
            if param_bounds[i][0] <= theta[i] <=param_bounds[i][1]:
                model = func(x, *theta)
                sigma2 = yerr ** 2
                return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
            else:
                return -1*np.inf
    else:
        model = func(x, *theta)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))      

def fit_function_mcmc(func, xdata, ydata, yerr, nwalkers=50, nsteps=10_000_000,intial_vals=None,param_bounds=None):
    ndim = func.__code__.co_argcount - 1  # number of parameters in the model function
    if intial_vals is None:
        p0 = np.random.rand(nwalkers, ndim)  # initial positions of the walkers in parameter space
    else:
        p0 = np.asarray(intial_vals) + (np.random.rand(nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda theta, x, y, yerr: log_likelihood(theta, x, y, yerr, func, param_bounds), args=(xdata, ydata, yerr))

    max_n = 10_000_000

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    samples = sampler.get_chain(discard=200, thin=15, flat=True)
    return samples

def report_fit_parameters(samples):
    medians = np.median(samples, axis=0)
    lower_quantiles = np.percentile(samples, 16, axis=0)
    upper_quantiles = np.percentile(samples, 84, axis=0)
    for i in range(samples.shape[1]):
        print(f"Parameter {i}: {round(medians[i],3)} + {round(upper_quantiles[i] - medians[i],3)} - {round(medians[i] - lower_quantiles[i],3)}")
        
    return round(medians[0],3),round(medians[1],3)


 