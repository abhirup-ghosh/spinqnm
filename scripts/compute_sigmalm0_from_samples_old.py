import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lal
import lalsimulation as lalsim
import compute_sigmalm0_SimIMREOBGenerateQNMFreqV2 as calcqnm
import scipy
import scipy.signal as ss
from scipy import interpolate
import sys
import scipy.ndimage.filters as filter
from optparse import OptionParser
import corner

# Module for confidence calculations
class confidence(object):
  def __init__(self, counts):
    # Sort in descending order in frequency
    self.counts_sorted = np.sort(counts.flatten())[::-1]
    # Get a normalized cumulative distribution from the mode
    self.norm_cumsum_counts_sorted = np.cumsum(self.counts_sorted) / np.sum(counts)
    # Set interpolations between heights, bins and levels
    self._set_interp()
  def _set_interp(self):
    self._length = len(self.counts_sorted)
    # height from index
    self._height_from_idx = interpolate.interp1d(np.arange(self._length), self.counts_sorted, bounds_error=False, fill_value=0.)
    # index from height
    self._idx_from_height = interpolate.interp1d(self.counts_sorted[::-1], np.arange(self._length)[::-1], bounds_error=False, fill_value=self._length)
    # level from index
    self._level_from_idx = interpolate.interp1d(np.arange(self._length), self.norm_cumsum_counts_sorted, bounds_error=False, fill_value=1.)
    # index from level
    self._idx_from_level = interpolate.interp1d(self.norm_cumsum_counts_sorted, np.arange(self._length), bounds_error=False, fill_value=self._length)
  def level_from_height(self, height):
    return self._level_from_idx(self._idx_from_height(height))
  def height_from_level(self, level):
    return self._height_from_idx(self._idx_from_level(level))

# gaussian filter of histogram
def gf(P):
  return filter.gaussian_filter(P, sigma=1.0)

if __name__ == '__main__':

  parser = OptionParser()
  parser.add_option("-p", "--post-loc", dest="post_loc", help="path to directory containing the cbcBayesPostProc posterior_samples.dat output")
  (options, args) = parser.parse_args()
  post_loc = options.post_loc

  # Read data
  data = np.genfromtxt(post_loc + '/posterior_samples.dat', names=True, dtype=None)
  m1, m2, a1z, a2z, domega220, dtau220 = data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtauinv220']

  N_bins = 51

  ########################################################
  ## Computing (domega220, dtau220) histogram
  ########################################################

  domega220_bins = np.linspace(min(domega220), max(domega220), N_bins)
  dtau220_bins = np.linspace(min(dtau220), max(dtau220), N_bins)

  domega220_intp = (domega220_bins[:-1] + domega220_bins[1:])/2.
  dtau220_intp = (dtau220_bins[:-1] + dtau220_bins[1:])/2.

  diff_domega220 = np.mean(np.diff(domega220_bins))
  diff_dtau220 = np.mean(np.diff(dtau220_bins))

  P_domega220dtau220, domega220_bins, dtau220_bins = np.histogram2d(domega220, dtau220, bins=(domega220_bins, dtau220_bins), normed=True)
  P_domega220dtau220 = P_domega220dtau220.T

  ########################################################
  ## Computing (omega220, tau220) histogram
  ########################################################

  lm = [2,2]

  omega220_GR_array = []
  tau220_GR_array = []

  omega220_modGR_array = []
  tau220_modGR_array = []

  for idx in range(len(data)):

	    # Compute GR values of (omega220, tau220) from (m1, m2, a1z, a2z) samples
	    omega220_GR, tau220_GR = calcqnm.get_sigmalm0SI_GR(m1[idx], m2[idx], a1z[idx], a2z[idx], lm)

	    # create (omega220/2pi, tau220) GR arrays
	    omega220_GR_array.append(omega220_GR[0]), tau220_GR_array.append(tau220_GR[0])

	    # Compute modGR values of (omega220, tau220) by including the fractional deviations (domega220, dtau220)
	    omega220_modGR, tau220_modGR = omega220_GR*(1. + domega220[idx]), tau220_GR*(1 + dtau220[idx])

	    # create (omega220/2pi, tau220) modGR arrays
	    omega220_modGR_array.append(omega220_modGR[0]), tau220_modGR_array.append(tau220_modGR[0])

  omega220_GR_bins = np.linspace(min(omega220_GR_array), max(omega220_GR_array), N_bins)
  tau220_GR_bins = np.linspace(min(tau220_GR_array), max(tau220_GR_array), N_bins)

  omega220_modGR_bins = np.linspace(min(omega220_modGR_array), max(omega220_modGR_array), N_bins)
  tau220_modGR_bins = np.linspace(min(tau220_modGR_array), max(tau220_modGR_array), N_bins)

  omega220_GR_intp = (omega220_GR_bins[:-1] + omega220_GR_bins[1:])/2.
  tau220_GR_intp = (tau220_GR_bins[:-1] + tau220_GR_bins[1:])/2.

  omega220_modGR_intp = (omega220_modGR_bins[:-1] + omega220_modGR_bins[1:])/2.
  tau220_modGR_intp = (tau220_modGR_bins[:-1] + tau220_modGR_bins[1:])/2.

  diff_omega220_GR = np.mean(np.diff(omega220_GR_bins))
  diff_tau220_GR = np.mean(np.diff(tau220_GR_bins))

  diff_omega220_modGR = np.mean(np.diff(omega220_modGR_bins))
  diff_tau220_modGR = np.mean(np.diff(tau220_modGR_bins))

  P_omega220_GRtau220_GR, omega220_GR_bins, tau220_GR_bins = np.histogram2d(omega220_GR_array, tau220_GR_array, bins=(omega220_GR_bins, tau220_GR_bins), normed=True)
  P_omega220_GRtau220_GR = P_omega220_GRtau220_GR.T

  P_omega220_modGRtau220_modGR, omega220_modGR_bins, tau220_modGR_bins = np.histogram2d(omega220_modGR_array, tau220_modGR_array, bins=(omega220_modGR_bins, tau220_modGR_bins), normed=True)
  P_omega220_modGRtau220_modGR = P_omega220_modGRtau220_modGR.T

  ########################################################
  ## Printing median values and credible levels of 1D histograms
  ########################################################

  # Marginalization to one-dimensional joint_posteriors
  P_omega220_modGR = np.sum(P_omega220_modGRtau220_modGR, axis=0) * diff_tau220_modGR
  P_tau220_modGR = np.sum(P_omega220_modGRtau220_modGR, axis=1) * diff_omega220_modGR

  

  ########################################################
  ## Plotting (domega220, dtau220) and (omega220, tau220)
  ########################################################
 
  conf_domega220dtau220 = confidence(P_domega220dtau220)
  s1_domega220dtau220 = conf_domega220dtau220.height_from_level(0.5)
  s2_domega220dtau220 = conf_domega220dtau220.height_from_level(0.9)

  conf_omega220_GRtau220_GR = confidence(P_omega220_GRtau220_GR)
  s1_omega220_GRtau220_GR = conf_omega220_GRtau220_GR.height_from_level(0.5)
  s2_omega220_GRtau220_GR = conf_omega220_GRtau220_GR.height_from_level(0.9)

  conf_omega220_modGRtau220_modGR = confidence(P_omega220_modGRtau220_modGR)
  s1_omega220_modGRtau220_modGR = conf_omega220_modGRtau220_modGR.height_from_level(0.5)
  s2_omega220_modGRtau220_modGR = conf_omega220_modGRtau220_modGR.height_from_level(0.9)

  plt.figure(figsize=(5,5))
  plt.scatter(domega220, dtau220, color='k', alpha=0.2, marker='.')
  plt.contour(domega220_intp, dtau220_intp, gf(P_domega220dtau220), levels=(s2_domega220dtau220,s1_domega220dtau220), linewidths=(1,1.5), colors='orange')
  plt.axvline(x=0, color='w', ls='--')
  plt.axhline(y=0, color='w', ls='--')
  plt.xlim([-1,2])
  plt.ylim([-1,2])
  plt.xlabel('$d\Omega_{220}$')
  plt.ylabel('$d\\tau^{-1}_{220}$')
  plt.tight_layout()
  plt.savefig(post_loc + '/qnmtest_dsigmalm0.png')

  plt.figure(figsize=(5,5))
  plt.contour(omega220_GR_intp, tau220_GR_intp, gf(P_omega220_GRtau220_GR), levels=(s2_omega220_GRtau220_GR,s1_omega220_GRtau220_GR), linewidths=(1,1.5), colors='orange')
  plt.contour(omega220_modGR_intp, tau220_modGR_intp, gf(P_omega220_modGRtau220_modGR), levels=(s2_omega220_modGRtau220_modGR,s1_omega220_modGRtau220_modGR), linewidths=(1,1.5), colors='k')
  #plt.contour(omega220_modGR_intp, tau220_modGR_intp, gf(P_omega220_modGRtau220_modGR), levels=(s2_omega220_modGRtau220_modGR,), linewidths=(1,1.5), colors='k')
  plt.xlabel('$\Omega_{220}$ Hz')
  plt.ylabel('$\\tau_{220}$ sec')
  plt.xlim([0, 200])
  plt.ylim([0, 0.05])
  plt.tight_layout()
  plt.savefig(post_loc + '/qnmtest_sigmalm0_m0p99.png')

  plt.figure(figsize=(5,5))
  plt.scatter(omega220_modGR_array, tau220_modGR_array, color='k', alpha=0.2, marker='.')
  plt.scatter(omega220_GR_array, tau220_GR_array, color='r', alpha=0.2, marker='.')
  plt.contour(omega220_modGR_intp, tau220_modGR_intp, gf(P_omega220_modGRtau220_modGR), levels=(s2_omega220_modGRtau220_modGR,s1_omega220_modGRtau220_modGR), linewidths=(1,1.5), colors='orange')
  #plt.contour(omega220_modGR_intp, tau220_modGR_intp, gf(P_omega220_modGRtau220_modGR), levels=(s2_omega220_modGRtau220_modGR,), linewidths=(1,1.5), colors='orange')
  plt.xlabel('$\Omega_{220}$ Hz')
  plt.ylabel('$\\tau_{220}$ sec')
  plt.xlim([0, 200])
  plt.ylim([0, 0.05])
  plt.tight_layout()
  plt.savefig(post_loc + '/qnmtest_sigmalm0_m0p99_scatter.png')

  samples = np.vstack((omega220_modGR_array, tau220_modGR_array)).T
  corner.corner(samples)
  plt.savefig(post_loc + '/qnmtest_sigmalm0_modGR_corner.png')
