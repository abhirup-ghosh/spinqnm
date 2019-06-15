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

  ##############################################################################
  ## Read (m1, m2, a1z, a2z, domega, dtau) posterior samples
  ##############################################################################

  data = np.genfromtxt(post_loc + '/posterior_samples.dat', names=True, dtype=None)
  m1, m2, a1z, a2z, domega, dtau = data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtauinv220']

  ##############################################################################
  ## Computing (omega, tau) GR and modGR posterior samples
  ##############################################################################

  lm = [2,2]

  omega_GR_array = []
  tau_GR_array = []

  omega_modGR_array = []
  tau_modGR_array = []

  for idx in range(len(data)):

	    # Compute GR values of (omega, tau) from (m1, m2, a1z, a2z) samples
	    omega_GR, tau_GR = calcqnm.get_sigmalm0SI_GR(m1[idx], m2[idx], a1z[idx], a2z[idx], lm)

	    # create (omega/2pi, tau) GR arrays
	    omega_GR_array.append(omega_GR[0]), tau_GR_array.append(tau_GR[0])

	    # Compute modGR values of (omega, tau) by including the fractional deviations (domega, dtau)
	    omega_modGR, tau_modGR = omega_GR*(1. + domega[idx]), tau_GR*(1 + dtau[idx])

	    # create (omega/2pi, tau) modGR arrays
	    omega_modGR_array.append(omega_modGR[0]), tau_modGR_array.append(tau_modGR[0])

  ##############################################################################
  ## Plotting
  ##############################################################################

  samples_domega_dtau = np.vstack((domega, dtau)).T
  corner.corner(samples_domega_dtau, labels=[r"$d\Omega$", r"$d\tau$"], quantiles=(0.16, 0.84), show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/qnmtest_frac_params_corner.png')

  samples_omega_tau_GR = np.vstack((omega_GR_array, tau_GR_array)).T
  corner.corner(samples_omega_tau_GR, labels=[r"$\Omega$(Hz)", r"$\tau$ (s)"], quantiles=(0.16, 0.84), show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/qnmtest_abs_params_GR_corner.png')

  samples_omega_tau_modGR = np.vstack((omega_modGR_array, tau_modGR_array)).T
  corner.corner(samples_omega_tau_modGR, labels=[r"$\Omega$(Hz)", r"$\tau$ (s)"], quantiles=(0.16, 0.84), show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/qnmtest_abs_params_modGR_corner.png')

