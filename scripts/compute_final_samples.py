import os
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
import imrtgrutils_final as tgr
import argparse

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

def Mjfinal220(omega220,tau220):

    jf=1-pow((omega220*tau220/2.-0.7)/(1.4187),-1/0.4990)
    Mf=((1.5251 - 1.1568*pow((1 - jf),0.1292))/omega220)*(pow(lal.C_SI,3)/lal.G_SI/lal.MSUN_SI)

    return Mf,jf

if __name__ == '__main__':

  parser = OptionParser()
  parser.add_option("-p", "--post-loc", dest="post_loc", help="path to directory containing the cbcBayesPostProc posterior_samples.dat output")
  parser.add_option("-o", "--outdir", dest="outdir", help="creates output directory inside the post_loc")
  parser.add_option("-i", "--inj_data", dest="inj_data", nargs=9, type="float", help="injection array")
  (options, args) = parser.parse_args()
  post_loc = options.post_loc
  outdir = options.outdir
  inj_data = options.inj_data

  print inj_data

  os.system('mkdir -p %s/%s'%(post_loc, outdir))

  ##############################################################################
  ## Injection
  ##############################################################################

  lm = [2,2]

  if inj_data != None:
	m1, m2, a1, a2, a1z, a2z, tilt1, tilt2, phi12 = inj_data
	mf_inj, af_inj = tgr.calc_final_mass_spin(m1, m2, a1, a2, a1z, a2z, tilt1, tilt2, phi12, 'bbh_average_fits_precessing')
	omega_inj, tau_inj = calcqnm.get_sigmalm0SI_GR(np.array([m1]), np.array([m2]), np.array([a1z]), np.array([a2z]), lm)
  	freq_inj = omega_inj/(2.*np.pi)

  else:
	freq_inj, tau_inj, mf_inj, af_inj = 0., 0., 0., 0.

  ##############################################################################
  ## Read (m1, m2, a1z, a2z, domega, dtau) posterior samples
  ##############################################################################

  data = np.genfromtxt(post_loc + '/posterior_samples.dat', names=True, dtype=None)
  if ('mf_evol' in data.dtype.names) and ('af_evol' in data.dtype.names):
  	mf, af, m1, m2, a1z, a2z, domega, dtau = data['mf_evol'], data['af_evol'], data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtau220']
  elif ('mf' in data.dtype.names) and ('af' in data.dtype.names):
        mf, af, m1, m2, a1z, a2z, domega, dtau = data['mf'], data['af'], data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtau220']
  else:
	m1, m2, a1, a2, a1z, a2z, domega, dtau = data['mass_1'], data['mass_2'], data['a_1'], data['a_2'], data['chi_1'], data['chi_2'], data['domega220'], data['dtau220']
	tilt1, tilt2, phi12 = np.zeros(len(m1)), np.zeros(len(m1)), np.zeros(len(m1))
	mf, af = tgr.calc_final_mass_spin(m1, m2, a1, a2, a1z, a2z, tilt1, tilt2, phi12, 'bbh_average_fits_precessing')

  ##############################################################################
  ## Computing (omega, tau) GR and modGR posterior samples
  ##############################################################################

  lm = [2,2]

  # create (omega, tau) GR and modGRarrays
  omega_GR, tau_GR = calcqnm.get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm)
  freq_GR = omega_GR/(2.*np.pi)

  omega_modGR, tau_modGR = calcqnm.get_sigmalm0SI_modGR(omega_GR, tau_GR, domega, dtau)
  freq_modGR = omega_modGR/(2.*np.pi)

  mf_modGR, af_modGR = Mjfinal220(freq_modGR*2*np.pi,tau_modGR)

  ##############################################################################
  ## Saving data
  ##############################################################################
  np.savetxt(post_loc + '/%s/qnmtest_frac_params_data.dat'%outdir, np.c_[domega, dtau], header='domega dtau')
  np.savetxt(post_loc + '/%s/qnmtest_abs_params_omega_tau_GR_data.dat'%outdir, np.c_[omega_GR, tau_GR], header='omega_GR tau_GR')
  np.savetxt(post_loc + '/%s/qnmtest_abs_params_omega_tau_modGR_data.dat'%outdir, np.c_[omega_modGR, tau_modGR], header='omega_modGR tau_modGR')
  np.savetxt(post_loc + '/%s/qnmtest_abs_params_f_tau_GR_data.dat'%outdir, np.c_[freq_GR, tau_GR], header='freq_GR tau_GR')
  np.savetxt(post_loc + '/%s/qnmtest_abs_params_f_tau_modGR_data.dat'%outdir, np.c_[freq_modGR, tau_modGR], header='freq_modGR tau_modGR')
  np.savetxt(post_loc + '/%s/qnmtest_mf_af_GR_data.dat'%outdir, np.c_[mf, af], header='mf af')
  np.savetxt(post_loc + '/%s/qnmtest_mf_af_modGR_data.dat'%outdir, np.c_[mf_modGR, af_modGR], header='mf_modGR af_modGR')

  ##############################################################################
  ## Plotting
  ##############################################################################

  samples_domega_dtau = np.vstack((domega, dtau)).T
  corner.corner(samples_domega_dtau, labels=[r"$d\Omega$", r"$d\tau$"], quantiles=(0.16, 0.5, 0.84), truths=[0,0], truth_color='g', show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/qnmtest_frac_params_corner.png'%outdir)

  samples_omega_tau_GR = np.vstack((omega_GR, tau_GR*1000.)).T
  corner.corner(samples_omega_tau_GR, labels=[r"$\Omega$(Hz)", r"$\tau$ (ms)"], quantiles=(0.16, 0.5, 0.84), truths=[freq_inj*2.*np.pi, tau_inj], show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/qnmtest_abs_params_omega_tau_GR_corner.png'%outdir)

  samples_omega_tau_modGR = np.vstack((omega_modGR, tau_modGR*1000.)).T
  corner.corner(samples_omega_tau_modGR, labels=[r"$\Omega$(Hz)", r"$\tau$ (ms)"], quantiles=(0.16, 0.5, 0.84), truths=[freq_inj*2.*np.pi, tau_inj], show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/qnmtest_abs_params_omega_tau_modGR_corner.png'%outdir)

  samples_freq_tau_GR = np.vstack((freq_GR, tau_GR*1000.)).T
  corner.corner(samples_freq_tau_GR, labels=[r"$f$(Hz)", r"$\tau$ (ms)"], quantiles=(0.16, 0.5, 0.84), truths=[freq_inj, tau_inj], show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/qnmtest_abs_params_f_tau_GR_corner.png'%outdir)

  samples_freq_tau_modGR = np.vstack((freq_modGR, tau_modGR*1000.)).T
  corner.corner(samples_freq_tau_modGR, labels=[r"$f$(Hz)", r"$\tau$ (ms)"], quantiles=(0.16, 0.5, 0.84), truths=[freq_inj, tau_inj], show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/qnmtest_abs_params_f_tau_modGR_corner.png'%outdir)

  samples_mf_af_modGR = np.vstack((mf_modGR, af_modGR)).T
  corner.corner(samples_mf_af_modGR, labels=["$M_f$", "$a_f$"], quantiles=(0.16, 0.5, 0.84), truths=[mf_inj, af_inj], show_titles=True, title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/qnmtest_mf_af_modGR_corner.png'%outdir)

  samples_domega_dtau = np.vstack((domega, dtau, freq_GR, tau_GR*1000., mf, af)).T
  corner.corner(samples_domega_dtau, labels=[r"$df_{220}$", r"$d\tau_{220}$",r"$f_{220,GR}$",r"$\tau_{220,GR}$",r"$M_{f,GR}$",r"$\chi_{f,GR}$"], show_titles=True, truths=[0., 0., freq_inj, tau_inj, mf_inj, af_inj], title_kwargs={"fontsize": 12})
  plt.savefig(post_loc + '/%s/corner6D.png'%outdir)
