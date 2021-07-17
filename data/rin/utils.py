# -*- coding: utf-8 -*-
#
#       Copyright 2020
#       Maximiliano Isi <max.isi@ligo.org>
#       Ben Farr ... [add me]
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import os
import numpy as np
from scipy.stats import gaussian_kde 
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import pickle as pkl
from pylab import *
import scipy.stats as ss
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde as Bounded_1d_kde

# ############################################################################
# rcParams

# make plots fit the LaTex column size but rescale them for ease of display
scale_factor = 2

# Get columnsize from LaTeX using \showthe\columnwidth
fig_width_pt = scale_factor*246.0
# Convert pts to inches
inches_per_pt = 1.0/72.27               
# Golden ratio
fig_ratio = (np.sqrt(5)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
fig_height =fig_width*fig_ratio

figsize_column = (fig_width, fig_height)
figsize_square = (fig_width, fig_width)

fig_width_page = scale_factor*inches_per_pt*508.87
figsize_page = (fig_width_page, fig_height)

rcParams = {'figure.figsize': figsize_column}

# LaTex text font sizse in points (rescaled as above)
fs = scale_factor*9
fs_label = 0.8*fs
rcParams['axes.labelsize'] = fs
rcParams['legend.fontsize'] = fs
rcParams['xtick.labelsize'] = fs_label
rcParams['ytick.labelsize'] = fs_label
rcParams["text.usetex"] = "true"


# ############################################################################
# NAMING

# Map catalog id to super-event id
catalog_id_to_superevent_id_dict = {
    "GW190408A" : "S190408an",
    "GW190412A" : "S190412m",
    "GW190413A" : "S190413i",
    "GW190413B" : "S190413ac",
    "GW190421A" : "S190421ar",
    "GW190424A" : "S190424ao",
    "GW190425A" : "S190425z",
    "GW190426A" : "S190426c",
    "GW190503A" : "S190503bf",
    "GW190512A" : "S190512at",
    "GW190513A" : "S190513bm",
    "GW190514A" : "S190514n",
    "GW190517A" : "S190517h",
    "GW190519A" : "S190519bj",
    "GW190521A" : "S190521g",
    "GW190521B" : "S190521r",
    "GW190527A" : "S190527w",
    "GW190602A" : "S190602aq",
    "GW190620A" : "S190620e",
    "GW190630A" : "S190630ag",
    "GW190701A" : "S190701ah",
    "GW190706A" : "S190706ai",
    "GW190707A" : "S190707q",
    "GW190708A" : "S190708ap",
    "GW190719A" : "S190719an",
    "GW190720A" : "S190720a",
    "GW190727A" : "S190727h",
    "GW190728A" : "S190728q",
    "GW190731A" : "S190731aa",
    "GW190803A" : "S190803e",
    "GW190814A" : "S190814bv",
    "GW190828A" : "S190828j",
    "GW190828B" : "S190828l",
    "GW190909A" : "S190909w",
    "GW190910A" : "S190910s",
    "GW190915A" : "S190915ak",
    "GW190924A" : "S190924h",
    "GW190929A" : "S190929d",
    "GW190930A" : "S190930s",
    "GW150914A" : "S150914",
    "GW151012A" : "S151012",
    "GW151226A" : "S151226",
    "GW170104A" : "S170104",
    "GW170608A" : "S170608",
    "GW170729A" : "S170729",
    "GW170809A" : "S170809",
    "GW170814A" : "S170814",
    "GW170817A" : "S170817",
    "GW170818A" : "S170818",
    "GW170823A" : "S170823",
    "S191109d":"S191109d", 
    "S200112r":"S200112r", 
    "S200129m":"S200129m", 
    "S200208q":"S200208q",         
    "S200224ca":"S200224ca", 
    "S200311bg":"S200311bg"
}

catalog_ids = sorted(catalog_id_to_superevent_id_dict.keys())
superevent_ids = [catalog_id_to_superevent_id_dict[v] for v in catalog_ids]

# Build the inverse mapping: from super-event id to catalog id
superevent_id_to_catalog_id_dict = dict(zip(superevent_ids, catalog_ids))

# Two handy routines
def cid_to_sid(catalog_id):
    """ Get GraceDB superevent ID from catalog ID.
    """
    return catalog_id_to_superevent_id_dict[catalog_id]

def sid_to_cid(superevent_id):
    """ Get catalog ID from GraceDB superevent ID.
    """
    return superevent_id_to_catalog_id_dict[superevent_id]

o1o2_events = [k for k in catalog_id_to_superevent_id_dict.keys()
               if 'GW19' not in k]

# explicitly list events considerd in this paper
all_events = [
    'GW150914A', 'GW151226A', 'GW170104A', 'GW170608A', 'GW170809A',
    'GW170814A', 'GW170818A', 'GW170823A', 'GW190408A', 'GW190412A',
    'GW190421A', 'GW190425A', 'GW190503A', 'GW190512A', 'GW190513A',
    'GW190517A', 'GW190519A', 'GW190521A', 'GW190521B', 'GW190602A',
    'GW190630A', 'GW190706A', 'GW190707A', 'GW190708A', 'GW190720A',
    'GW190727A', 'GW190728A', 'GW190814A', 'GW190828A', 'GW190828B',
    'GW190910A', 'GW190915A', 'GW190924A',
]


# ############################################################################
# EVENT UTILS

# get module path
base_path = os.path.dirname(os.path.realpath(__file__))

# load style properties from GWTC-2
with open(os.path.join(base_path, 'colors.pkl'), 'rb') as f:
    _STYLE_DICT = pkl.load(f)

# load PE properties from GWTC-2
with open(os.path.join(base_path, 'parameters.pkl'), 'rb') as f:
    _PE_DICT = pkl.load(f)

# load PE properties from GWTC-1
with open(os.path.join(base_path, 'parameters_o1o2.pkl'), 'rb') as f:
    _PE_DICT_O1O2 = pkl.load(f)
    for k,v in _PE_DICT.items():
        if k in _PE_DICT_O1O2:
            v.update(_PE_DICT_O1O2[k])
    for k,v in _PE_DICT_O1O2.items():
        if k not in _PE_DICT:
            _PE_DICT[k] = v

_NAME_DICT = {v: k for k,v in _PE_DICT['NAME'].items()}

_pesummary_names = {
  'MC': 'chirp_mass',
  'M': 'total_mass',
  'Q': 'mass_ratio',
  'M1': 'mass_1',
  'M2': 'mass_2',
  'MF': 'final_mass',
  'CHIF': 'final_spin',
  'CHI1': 'spin_1',
  'CHI2': 'spin_2',
  'CHIEFF': 'chi_eff',
  'CHIP': 'chi_p',
  'Z': 'redshift',
  'D': 'luminosity_distance',
  'SNR': 'networkmatchedfiltersnr',
}
for k,v in _pesummary_names.copy().items():
    if 'mass' in v and 'ratio' not in v:
        _pesummary_names[k+'_SRC'] = v+'_source'

# from catalog
column_name2tex_name = {
        'total_mass_source': r'M/M_\odot',
        'chirp_mass_source': r'\mathcal{{M}}/M_\odot',
        'total_mass': r'(1+z)M/M_\odot',
        'chirp_mass': r'(1+z)\mathcal{{M}}/M_\odot',
        'mass_1_source': r'm_1/M_\odot',
        'mass_2_source': r'm_2/M_\odot',
        'mass_1': r'(1+z)m_1/M_\odot',
        'mass_2': r'(1+z)m_2/M_\odot',
        'spin_1': r'\chi_1',
        'spin_2': r'\chi_2',
        'chi_eff':r'\chi_{{\rm eff}}',
        'chi_p':r'\chi_p',
        'luminosity_distance':r'D_L/{\rm Gpc}',
        'redshift':r'z',
        'mass_ratio':r'q',
        'final_mass_source': r'M_{\rm f}/M_\odot',
        'final_mass': r'(1+z)M_{\rm f}/M_\odot',
        'final_mass_source': r'M_{\rm f}^{\rm det}/M_\odot',
        'final_spin': r'\chi_{\rm f}',
        'networkmatchedfiltersnr': r'{\rm SNR}',
}

class Parameter(object):
    def __init__(self, name):
        self.name = name.upper()
        self.pe_name = _pesummary_names[self.name]
        self.macro_key = self.pe_name.replace('_', '')
        if 'mass' in self.macro_key and 'ratio' not in self.macro_key:
            if 'source' not in self.macro_key:
                self.macro_key += 'det'
        self.latex = column_name2tex_name[self.pe_name]

class Event(object):
    def __init__(self, key):
        match_o1o2 = [k.strip('GW').strip('A') in key for k in o1o2_events]
        self.run = 'O3a'
        if any(match_o1o2):
            # this is an event from O1O2
            self.cid = [k for k,m in zip(o1o2_events,match_o1o2) if m][0]
            self.sid = cid_to_sid(self.cid)
            self.run = 'O1O2'
        elif key in superevent_ids:
            self.sid = key
            self.cid = sid_to_cid(self.sid)
        elif key.upper() in catalog_ids:
            self.cid = key.upper()
            self.sid = cid_to_sid(self.cid)
        else:
            # try to guess whether the key is mixed up
            new_key = key.replace('GW', 'S')
            if new_key in superevent_ids:
                self.sid = new_key
                self.cid = sid_to_cid(self.sid)
            elif key in _NAME_DICT:
                self.cid = _NAME_DICT[key]
                self.sid = cid_to_sid(self.cid)
            else:
                raise ValueError("unrecognized event %r" % key)
        self.in_paper = self.cid in all_events
        # alternative capitalization, e.g. 'GW190930A' -> 'GW190930a'
        alt_cid = self.cid[:-1] + self.cid[-1].lower()
         # style properties
        if self.run == 'O3a':
            self.color = _STYLE_DICT['colors'][alt_cid]
            self.ls = _STYLE_DICT['linestyles'][alt_cid]
            self.lw = _STYLE_DICT['linewidths'][alt_cid]
        else:
            self.color = ''
            self.ls = '-'
            self.lw = 1
        # parameters
        self.pe = {k: v.get(self.cid) for k,v in _PE_DICT.items()}
        self.ifos = self.pe['OBSERVINGINSTRUMENTS']
        self.name = self.pe['NAME']

    def get_param(self, key):
        return self.pe[Parameter(key).macro_key]

    @property
    def parameters(self):
        keys = []
        for k in _pesummary_names.keys():
            if Parameter(k).macro_key in self.pe:
                keys.append(k.lower())
        return keys
    
    @property
    def name_macro(self):
        if 'GW19' in self.cid:
            return r'\NAME{%s}' % self.cid
        else:
            return self.name


# ############################################################################
# STATS

def multiply_likelihood_kdes(all_kdes, x):
    """ Multiply a set of KDEs, evaluated over array.

    Arguments
    ---------
    all_kdes: list
        list of KDE functions to be combined.
    x: array
        array of parameter values over which to evaluate KDEs.

    Returns
    -------
    joint_like: array
        combined KDE evaluated over `x` grid.
    """
    dx = x[1] - x[0]
    joint_like = np.ones_like(x)
    for kde in all_kdes:
        joint_like *= kde(x)
        joint_like /= np.sum(joint_like*dx)
    return joint_like

def multiply_likelihoods(all_samples, range=None, nbins=1000,
                         kde=gaussian_kde, **kwargs):
    """ Multiply PDFs starting from samples.

    Arguments
    ---------
    all_samples: list
        list of arrays of samples drawn from the PDFs to be combined.
    range: tuple, None
        range of x over which to combine PDF(x), defaults to the min and max
        over all sets of samples provided.
    nbins: int
        number of bins used to produce grid overwhich to multiply PDFs
        (def. 1000).
    kde: function
        function to prodce KDE from samples (def. scipy.stats.gaussian_kde).

    kwargs:
        additional arguments passed to KDE function.

    Returns
    -------
    joint_like: array
        combined PDF evaluated over `x` grid.
    x: array
        parameter grid over which PDF was evaluated.
    """
    if range is None:
        xmin = min([min(s) for s in all_samples])
        xmax = max([max(s) for s in all_samples])
    else:
        xmin, xmax = range
    x = np.linspace(xmin, xmax, nbins)
    all_kdes = []
    for samples in all_samples:
        pdf = kde(samples, **kwargs)
        all_kdes.append(interp1d(x, pdf(x)))
    joint_like = multiply_likelihood_kdes(all_kdes, x)
    return joint_like, x

def get_sym_interval_from_pdf(y, x, p=0.9, normalize=True):
    """ Compute symmetric credible interval from PDF evaluated over grid.

    Arguments
    ---------
    y: array
        array of PDF values
    x: array
        array of parameter values corresponding to `y`.
    p: float
        credible level (def. 0.9).
    normalize: bool
        normalize PDF before computing CI (def. True).

    Returns
    -------
    left: float
        parameter value corresponding to lower interval edge.
    right: float
        parameter value corresponding to upper interval edge.
    """
    cdf_left = 0.5*(1 - p)
    cdf_right = 1 - cdf_left
    # get CDF from PDF
    cdf = cumtrapz(y, x, initial=0.)
    if normalize:
        cdf /= cdf.max()
    # compute lower edge
    if cdf_left < min(cdf):
        print("WARNING: value below range.")
        left = min(x)
        cdf_interp = None
    else:
        cdf_interp = interp1d(cdf, x)
        left = cdf_interp(cdf_left)
    # compute upper edge
    if cdf_right > max(cdf):
        print("WARNING: value above range.")
        right = max(x)
    else:
        cdf_interp = cdf_interp or interp1d(cdf, x)
        right = cdf_interp(cdf_right)
    return left, right

def get_ul_from_pdf(y, x, p=0.9, normalize=True):
    """ Compute symmetric upper limit from PDF evaluated over grid.

    Arguments
    ---------
    y: array
        array of PDF values
    x: array
        array of parameter values corresponding to `y`.
    p: float
        credible level (def. 0.9).
    normalize: bool
        normalize PDF before computing UL (def. True).

    Returns
    -------
    ul: float
        parameter value corresponding to UL.
    """
    # get CDF from PDF
    cdf = cumtrapz(y, x, initial=0.)
    if normalize:
        cdf /= cdf.max()
    # compute UL
    if p > max(cdf):
        print("WARNING: value above range.")
        ul = max(x)
    else:
        cdf_interp = interp1d(cdf, x)
        ul = cdf_interp(p)
    return ul

def get_negpos_uls_from_pdf(y, x, p=0.9):
    """ Get p-credible upper limit for of absolute value of negative/positive
    samples treated separately.
    
    Arguments
    ---------
    y: array
        PDF(x) evaluated on x grid.
    x: array
        x values.
    p: float
        credibility (def. 0.9).
    
    Returns
    -------
    ul_m : float
        limit for negative values.
    ul_p : float
        limit for positive values.
    """
    ul_p =  get_ul_from_pdf(y[x>0], x[x>0], normalize=True, p=p)
    ul_m = -get_ul_from_pdf(y[x<0], x[x<0], normalize=True, p=1-p)
    return ul_m, ul_p

def get_negpos_uls_from_samples(samples, p=0.9):
    """ Get p-credible upper limit for of absolute value of negative/positive
    samples treated separately.
    
    Arguments
    ---------
    y: array
        PDF(x) evaluated on x grid.
    x: array
        x values.
    p: float
        credibility (def. 0.9).
    
    Returns
    -------
    ul_m : float
        limit for negative values.
    ul_p : float
        limit for positive values.
    """
    samples_p = samples[samples > 0]
    ul_p = np.percentile(samples_p, p*100)
    samples_m = samples[samples < 0]
    ul_m = np.percentile(samples_m, (1-p)*100)
    return ul_m, ul_p

def get_quantile_from_pdf(y, x, target=0, normalize=True):
    """ Get one-sided quantile of target value from PDF data.
    
    Arguments
    ---------
    y: array
        PDF(x) evaluated on x grid.
    x: array
        x values.
    target: float
        x value for which to compute quantile (def. 0).
    normalize: bool
        whether to normalize PDF before computing quantile (def. True).
    
    Returns
    -------
    q: float
       quantile at target. 
    """
    # get CDF from PDF
    cdf = cumtrapz(y, x, initial=0.)
    if normalize:
        cdf /= cdf.max()
    # interpolate
    cdf_interp = interp1d(x, cdf)
    return cdf_interp(target)

def get_quantile_from_samples(x, target=0, twosided=False):
    """ Get quantile (one or two-sided) of target value from samples.
    
    Arguments
    ---------
    x: array
        array of samples.
    target: float
        x value for which to compute quantile (def. 0).
    twosided: bool
        whether to compute two-sided quantile (def. False).
    
    Returns
    -------
    q: float
       quantile at target. 
    """
    if twosided:
        abs_dist = np.abs(x - np.median(x))
        target_dist = np.abs(target - np.median(x))
        q = len(abs_dist[abs_dist <= target_dist]) / len(x)
    else:
        q =len(x[x <= target])/len(x)
    return q

def get_2d_quantile_from_samples(x, y, target=(0, 0), reflect_around_x=True):
    """ Get 2D quantile of target value from samples (x, y).
    
    The 2D quantile is defined as the isoprobability contour that passes
    through the target point.
    
    WARNING: this function is written with the hyperparameters (x=mu, y=sigma)
    in mind, for which the usual target (0, 0) lies at the edge of the sigma 
    prior. To deal with this, the 2D distribution is reflected around the 
    x-axis by default (behavior controlled by `reflect_around_x` argument).
    
    Arguments
    ---------
    x: array
        x samples (1D).
    y: array
        y samples (1D).
    target: array
        coordinates at which to evaluate quantile (def. [0, 0]).
    reflect_around_x: bool
        whether to reflect points around the x-axis (def. True).
        
    Returns
    -------
    q: float
        quantile at target.
    """
    pts = np.row_stack((x, y))
    if reflect_around_x:
        pts_reflected = np.column_stack((pts, np.row_stack((x, -y))))
    else:
        pts_reflected = pts
    kde = gaussian_kde(pts_reflected)
    pts_densities = kde(pts)
    target_density = kde(target)
    q = np.count_nonzero(target_density < pts_densities) / len(pts_densities)
    return q
    
def draw_samples(pdf, x_min, x_max, nsamp=1000, **kwargs):
    """ Make `nsamp` draws from PDF using rejection sampling.

    Arguments
    ---------
    pdf: function
        probability density function for quantity x.
    x_min: float
        minimum value of x.
    x_max: float
        maximum value of x.
    nsamp: int
        number of samples to draw through rejection sampling..

    Returns
    -------
    samples: array
        array of samples drawn from PDF.
    """
    p_max = kwargs.pop('p_max', None)
    if p_max is None:
        # guess maximum of PDF
        x = np.linspace(x_min, x_max, 1000)  
        p_max = max(pdf(x))
    # rejection sampling
    sample_list = []
    for i in range(nsamp):
        sample = np.random.uniform(x_min, x_max) # 10*x_max
        while pdf(sample) < np.random.uniform(0, p_max):
            sample = np.random.uniform(x_min, x_max)
        sample_list.append(sample)
    samples = np.array(sample_list)
    return samples

"""
The following routine, Bounded_2d_kde, was copied from
https://git.ligo.org/publications/gw190412/gw190412-discovery/-/blob/851f91431b7c36e7ea66fa47e8516f2aef9d7daf/scripts/bounded_2d_kde.py
"""

"""
A bounded 2-D KDE class for all of your bounded 2-D KDE needs.
"""
from scipy.stats import gaussian_kde as kde

class Bounded_2d_kde(kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, y])

        if self.xhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, y])

        if self.ylow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.ylow - y])

        if self.yhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.yhigh - y])

        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.ylow - y])

            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.yhigh - y])

        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results


# ############################################################################
# PLOTTING

def kdeplot_2d_clevels(xs, ys, levels=11, **kwargs):
    """ Plot contours at specified credible levels.

    Arguments
    ---------
    xs: array
        samples of the first variable.
    ys: array
        samples of the second variable, drawn jointly with `xs`.
    levels: float, array
        if float, interpreted as number of credible levels to be equally 
        spaced between (0, 1); if array, interpreted as list of credible
        levels.
    xlow: float
        lower bound for abscissa passed to Bounded_2d_kde (optional).
    xigh: float
        upper bound for abscissa passed to Bounded_2d_kde (optional).
    ylow: float
        lower bound for ordinate passed to Bounded_2d_kde (optional).
    yhigh: float
        upper bound for ordinate passed to Bounded_2d_kde (optional).
    ax: Axes
        matplotlib axes on which to plot (optional).
    kwargs:
        additional arguments passed to plt.contour().
    """
    try:
        len(levels)
        f = 1 - np.array(levels)
    except TypeError:
        f = linspace(0, 1, levels+2)[1:-1]
    kde_kws = {k: kwargs.pop(k, None) for k in ['xlow', 'xhigh', 'ylow', 'yhigh']}
    k = Bounded_2d_kde(np.column_stack((xs, ys)), **kde_kws)
    size = max(10*(len(f)+2), 500)
    c = np.random.choice(len(xs), size=size)
    p = k(np.column_stack((xs[c], ys[c])))
    i = argsort(p)
    l = array([p[i[int(round(ff*len(i)))]] for ff in f])

    Dx = np.percentile(xs, 99) - np.percentile(xs, 1)
    Dy = np.percentile(ys, 99) - np.percentile(ys, 1)

    x = linspace(np.percentile(xs, 1)-0.1*Dx, np.percentile(xs, 99)+0.1*Dx, 128)
    y = linspace(np.percentile(ys, 1)-0.1*Dy, np.percentile(ys, 99)+0.1*Dy, 128)

    XS, YS = meshgrid(x, y, indexing='ij')
    ZS = k(np.column_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)

    ax = kwargs.pop('ax', gca())

    ax.contour(XS, YS, ZS, levels=l, **kwargs)
    
def kdeplot_2d_clevels_contourf(xs, ys, levels=11, **kwargs):
    """ Plot contours at specified credible levels.

    Arguments
    ---------
    xs: array
        samples of the first variable.
    ys: array
        samples of the second variable, drawn jointly with `xs`.
    levels: float, array
        if float, interpreted as number of credible levels to be equally 
        spaced between (0, 1); if array, interpreted as list of credible
        levels.
    xlow: float
        lower bound for abscissa passed to Bounded_2d_kde (optional).
    xigh: float
        upper bound for abscissa passed to Bounded_2d_kde (optional).
    ylow: float
        lower bound for ordinate passed to Bounded_2d_kde (optional).
    yhigh: float
        upper bound for ordinate passed to Bounded_2d_kde (optional).
    ax: Axes
        matplotlib axes on which to plot (optional).
    kwargs:
        additional arguments passed to plt.contour().
    """
    try:
        len(levels)
        f = 1 - np.array(levels)
    except TypeError:
        f = linspace(0, 1, levels+2)[1:-1]
    kde_kws = {k: kwargs.pop(k, None) for k in ['xlow', 'xhigh', 'ylow', 'yhigh']}
    k = Bounded_2d_kde(np.column_stack((xs, ys)), **kde_kws)
    size = max(10*(len(f)+2), 500)
    c = np.random.choice(len(xs), size=size)
    p = k(np.column_stack((xs[c], ys[c])))
    i = argsort(p)
    l = array([p[i[int(round(ff*len(i)))]] for ff in f])

    Dx = np.percentile(xs, 99) - np.percentile(xs, 1)
    Dy = np.percentile(ys, 99) - np.percentile(ys, 1)

    x = linspace(np.percentile(xs, 1)-0.1*Dx, np.percentile(xs, 99)+0.1*Dx, 128)
    y = linspace(np.percentile(ys, 1)-0.1*Dy, np.percentile(ys, 99)+0.1*Dy, 128)

    XS, YS = meshgrid(x, y, indexing='ij')
    ZS = k(np.column_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)

    ax = kwargs.pop('ax', gca())

    ax.contourf(XS, YS, ZS, levels=l, **kwargs)    

