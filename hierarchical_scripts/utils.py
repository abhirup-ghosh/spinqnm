import os
import re
from glob import glob
import numpy as np
from collections import OrderedDict
import json
import parse

# ----------------------------------------------------------------------------
# IO and other
# ----------------------------------------------------------------------------

def load_cache(param, cache_path):
    cache_exists = all(os.path.isfile(cache_path.format(par=param, hyper=k)) for
                       k in ['mu', 'sigma', 'pop'])
    if cache_exists:
        cache = {k: np.loadtxt(cache_path.format(par=param, hyper=k)) 
                 for k in ['mu', 'sigma', 'pop']}
    else:
        cache = {}
    return cache

def save_cache(fits, cache_path, paths=None):
    for param, fit in fits.items():
        try:
            post = fit.extract(['mu', 'sigma', 'pop'])
        except AttributeError:
            post = fit
        for k in ['mu', 'sigma', 'pop']:
            np.savetxt(cache_path.format(par=param, hyper=k), post[k])
        print("Cached %s"%  param)
    # record list of events
    if paths:
        cache_dir = os.path.dirname(cache_path)
        log_path = os.path.join(cache_dir, 'events.json')
        with open(log_path, 'w') as f:
            json.dump(paths, f, indent=4)
        print("Events logged: %r" % log_path)
        
def load_data(param, data_path, data_dict=None, path_dict=None):
    ''' Loads posteriors from individual event runs for a given parameter.
    
    Arguments
    ---------
    param: str
        parameter name
    data_path: str
        posterior sample path template, 'mypath/post_{par}_{event}.dat.gz'
    data_dict: dict
        dictionary to which to add loaded samples (opt.)
    path_dict: dict
        dictionary to which to add path information (opt.)
    
    Returns
    -------
    data_dict: dict
        dictionary with loaded samples.
    path_dict: dict
        dictionary with path information.
    '''
    if data_dict is None:
        data_dict = {}
    if path_dict is None:
        path_dict = {}
    paths = glob(data_path.format(par=param, event='*'))
    for path in paths:
        try:
            label = parse.parse(data_path, path)['event']
            data_dict[label] = np.loadtxt(path)
            path_dict[label] = path
        except AttributeError:
            pass
    return data_dict, path_dict

def load_data_fits(params, input_paths, cache=None, force_load_data=False, **kwargs):
    ''' Loads individual event posteriors from disk, or cached posteriors
    on hyperparameters, if cache is requested (and found).
    
    Arguments
    ---------
    params: list or str
        parameter name(s) to load
    input_paths: list or str
        posterior path template(s), e.g. 'mydir/posterior_{par}_{event}.dat.gz'
    cache: str
        path to cached hyperposteriors
    force_load_data: bool
        wheehter to load individual posteriors even if cache is found
    
    Returns
    -------
    data: dict
        Dictionary of dictionaries with samples for all parameters and events.
    fits: dict
        Dictionary of dictionaries with hyperparameter samples.
    paths: dict
        Dictionary of dictionaries with a record of where the posteriors where 
        loaded from for each individual event and parameter.
    '''
    if isinstance(params, str):
        params = [params]
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    data = OrderedDict({})
    fits = OrderedDict({})
    paths = OrderedDict({})
    for param in params:
        # load data if no cache found, or if for 
        cached = load_cache(param, cache) if cache else {}
        if not cached or force_load_data:
            dd, pd = {}, {}
            for path in input_paths:
                dd, pd = load_data(param, path, data_dict=dd, path_dict=pd, **kwargs)
            data[param] = dd
            paths[param] = pd
        if cached:
            fits[param] = cached
    return data, fits, paths


# ----------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------

def fit_it(chi_samples, nobs, nsamp, model, niter=2000):
    ''' Carry out Stan population fit for a given parameter ("chi").
    
    Arguments
    ---------
    chi_samples: array
        samples for all events `[[event_1_samples], [event_2_samples] ...]`
    nobs: int
        number of events (i.e. observations).
    nsamp: int
        number of samples per event (subselects for speed).
    model: pystan.StanModel
        stan model.
    niter: int
        number of stan iterations (def. 2000).
    '''
    #samples = [cs[:nsamp] for cs in chi_samples[:nobs]]
    chosen_chi_samples = chi_samples[np.random.choice(range(len(chi_samples)),
                                                      nobs, replace=False)]
    samples = []
    for cs in chosen_chi_samples:
        idxs = np.random.choice(range(len(cs)), nsamp, replace=False)
        samples.append(cs[idxs])
    stan_data = {'nobs': nobs, 'nsamp': nsamp, 'chis': samples}
    return model.sampling(data=stan_data, iter=niter)

def fit_all(data, fits=None, cache=None, max_nsamples=1000,
            scale_factors=None, **kwargs):
    ''' Fit a series of parameters, if fit not already in `fits`.
    
    Arguments
    ---------
    data: dict
        dictionary with individual-event samples for each parameter.
    fits: dict
        dictionary with population fit results.
    cache: str
        path template to cache fit results.
    max_nsamples: int
        maximum number of individual-event samples used in population fit.
    scale_factors: dict
        optionally scale parameters (x -> x/scale) to bring it into unit
        range; the keys must be names of parameters to rescale. (The scaling
        is undone before returning the result.)
        
    Returns
    -------
    fits: dict
        dictionary with fit results.
    '''
    if fits is None:
        fits = {}
    if scale_factors is None:
        scale_factors = {k: 1 for k in data.keys()}
    failed = []
    for label, chi_samples_dict in data.items():
        if label not in fits:
            chi_samples = np.array(list(chi_samples_dict.values()))
            # rescale samples by some scalar to ease sampling
            sf = scale_factors.get(label, 1.0)
            chi_samples /= sf
            # number of observations (events)
            no = len(chi_samples_dict)
            # number of posterior samples
            nsamples = [len(s) for s in chi_samples]
            ns = min(min(nsamples), max_nsamples)
            if no > 0:
                fit = fit_it(chi_samples, no, ns, **kwargs)
                # restore original scale to the data
                fits[label] = {k: fit[k]*sf for k in ['mu', 'sigma', 'pop']}
            else:
                print("WARNING: no data for %s" % label)
                failed.append(label)
    return fits

def draw_population(fits, params=None):
    """Extract PPD from fit, or produce it manually from hyperparameters.
    
    Arguments
    ---------
    fits: list
        list of Stan fits, or other hashable objects with samples.
    params: list
        list of parameter names (opt.)
    
    Returns
    -------
    draws: array
        population samples.
    """
    draws = OrderedDict({})
    if params is None:
        params = fits.keys()
    for i, param in enumerate(params):
        try:
            # check whether the population was already 
            # pre-sampled by stan during runtime
            draws[param] = fits[param]['pop']
        except (ValueError, KeyError):
            # if 'pop' was not produced (or saved) as
            # part of the fit, produce it directly
            mus = fits[param]['mu']
            sigmas = fits[param]['sigma']
            samples = []
            for mu,sigma in zip(mus, sigmas):
                samples.append(np.random.normal(mu, sigma))
            draws[param] = samples
    return draws

# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def ComputeSymCIedges(samples, ci=0.9):
    """ Symmetric CI.
    """
    lo = np.percentile(samples, 100.0 * (1-ci)*0.5)
    hi = np.percentile(samples, 100.0 * (1-(1-ci)*0.5))
    return hi, lo

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

def get_pop_summary(draws, p=0.9, scale_factors=None):
    ci_dict = {}
    scale_factors = scale_factors or {}
    print("Symmetric {}%-credible interval".format(int(p*100)))
    print("--------------------------------")
    for param, samples in draws.items():
        s = scale_factors.get(param, 1)
        ci1, ci2 = ComputeSymCIedges(np.array(samples)/s, p)
        med = np.median(np.array(samples)/s)
        ci_dict[param] = (med, ci1, ci2)
        print('%s (x%i):\t%.2f +%.2f -%.2f\t[%.2f]' % (param, 1/s, med, ci1-med, med-ci2, ci1-ci2))
    return ci_dict

def get_hyper_summary(fits, p=0.9, scale_factors=None):
    hyper_ci_dict = {}
    scale_factors = scale_factors or {}
    print("Hyperparameter constraints ({}% CL)".format(int(p*100)))
    print("----------------------------------")
    for param, fit in fits.items():
        s = scale_factors.get(param, 1)
        print(param, '(x%i)' % (1/s))
        # median
        mu_med = np.median(fit['mu']/s)
        mu_ci1, mu_ci2 = ComputeSymCIedges(fit['mu']/s, 0.90)
        # standard deviation
        sig_ul = np.percentile(fit['sigma']/s, 90)
        print('\tmu:\t%.2f +%.2f -%.2f  [%.2f]' % (mu_med, mu_ci1-mu_med, mu_med-mu_ci2, mu_ci1-mu_ci2))
        print('\tsigma: < %.2f' % sig_ul)
        hyper_ci_dict[param] = {'mu': (mu_med, mu_ci1, mu_ci2), 'sigma': sig_ul}
    return hyper_ci_dict