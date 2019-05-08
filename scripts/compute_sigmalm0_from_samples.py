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
  m1, m2, a1z, a2z, domega220, dtauinv220 = data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtauinv220']

  N_bins = 51

  ########################################################
  ## Computing (domega220, dtauinv220) histogram
  ########################################################

  domega220_bins = np.linspace(min(domega220), max(domega220), N_bins)
  dtauinv220_bins = np.linspace(min(dtauinv220), max(dtauinv220), N_bins)

  domega220_intp = (domega220_bins[:-1] + domega220_bins[1:])/2.
  dtauinv220_intp = (dtauinv220_bins[:-1] + dtauinv220_bins[1:])/2.

  P_domega220dtauinv220, domega220_bins, dtauinv220_bins = np.histogram2d(domega220, dtauinv220, bins=(domega220_bins, dtauinv220_bins), normed=True)
  P_domega220dtauinv220 = P_domega220dtauinv220.T

  ########################################################
  ## Computing (omega220, tau220) histogram
  ########################################################

  lm = [2,2]
  omega220 = []
  tau220 = []

  for idx in range(len(data)):

    # Compute GR values of (omega220, tauinv220) from (m1, m2, a1z, a2z) samples
    omega220_GR, tauinv220_GR = calcqnm.get_sigmalm0SI_GR(m1[idx], m2[idx], a1z[idx], a2z[idx], lm)

    # Compute modGR values of (omega220, tauinv220) by including the fractional deviations (domega220, dtauinv220)
    omega220_modGR, tauinv220_modGR = omega220_GR*(1. + domega220[idx]), tauinv220_GR*(1 + dtauinv220[idx])

    # Compute tau220 from tauinv220
    tau220_modGR = 1/tauinv220_modGR

    # create (omega220/2pi, tau220) arrays
    omega220.append(omega220_modGR[0]/(2.*np.pi)), tau220.append(tau220_modGR[0])

  omega220_bins = np.linspace(min(omega220), max(omega220), N_bins)
  tau220_bins = np.linspace(min(tau220), max(tau220), N_bins)

  omega220_intp = (omega220_bins[:-1] + omega220_bins[1:])/2.
  tau220_intp = (tau220_bins[:-1] + tau220_bins[1:])/2.

  P_omega220tau220, omega220_bins, tau220_bins = np.histogram2d(omega220, tau220, bins=(omega220_bins, tau220_bins), normed=True)
  P_omega220tau220 = P_omega220tau220.T

  ########################################################
  ## Plotting (domega220, dtauinv220) and (omega220, tau220)
  ########################################################
 
  conf_domega220dtauinv220 = confidence(P_domega220dtauinv220)
  s1_domega220dtauinv220 = conf_domega220dtauinv220.height_from_level(0.5)
  s2_domega220dtauinv220 = conf_domega220dtauinv220.height_from_level(0.9)

  conf_omega220tau220 = confidence(P_omega220tau220)
  s1_omega220tau220 = conf_omega220tau220.height_from_level(0.5)
  s2_omega220tau220 = conf_omega220tau220.height_from_level(0.9)

  plt.figure(figsize=(5,5))
  plt.contour(domega220_bins[:-1], dtauinv220_bins[:-1], gf(P_domega220dtauinv220), levels=(s2_domega220dtauinv220,s1_domega220dtauinv220), linewidths=(1,1.5), colors='orange')
  plt.xlim([-1,1])
  plt.ylim([-1,1])
  plt.savefig(post_loc + '/qnmtest_dsigmalm0.png')

  plt.figure(figsize=(5,5))
  plt.contour(omega220_bins[:-1], tau220_bins[:-1], gf(P_omega220tau220), levels=(s2_omega220tau220,s1_omega220tau220), linewidths=(1,1.5), colors='orange')
  plt.xlim([0, 400])
  plt.ylim([0, 400])
  plt.savefig(post_loc + '/qnmtest_sigmalm0.png')

