import numpy as np
import lal
import lalsimulation as lalsim

###########################################################
# function to compute GR value of sigmalm0 (in SI units)
# for a single set of m1, m2, a1z, a2z
###########################################################
def get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm):

  M = m1 + m2

  omegalm0SI = []
  taulm0SI = []

  for idx in range(len(M)):
    sigmalm0SI = lal.CreateCOMPLEX16Vector(1)
    lalsim.SimIMREOBGenerateQNMFreqV2(sigmalm0SI, m1[idx], m2[idx], np.array([0., 0., a1z[idx]]), np.array([0., 0., a2z[idx]]), lm[0], lm[1], 1, lalsim.SEOBNRv4)

    omegalm0SI.append(np.real((sigmalm0SI.data)[0]))
    taulm0SI.append(1./np.imag((sigmalm0SI.data)[0]))

  return np.asarray(omegalm0SI), np.asarray(taulm0SI)

###########################################################
# function to compute modGR values of sigmalm0 (in SI units)
###########################################################
def get_sigmalm0SI_modGR(omega_GR, tau_GR, domega, dtau):
  return omega_GR*(1. + domega), tau_GR*(1 + dtau)

if __name__ == '__main__':

  post_loc = '/home/abhirup.ghosh/Documents/Work/O3/2019/May/21/1242442967p4500/G333631/lalinference/20190525_pSEOBNRv4HM_domega220_dtauinv220/cbcBayes/posterior_samples.dat'
  data = np.genfromtxt(post_loc, names=True, dtype=None)
  m1, m2, a1z, a2z, domega220, dtauinv220 = data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtauinv220']

  lm_list = [[2,2],[2,1],[3,3],[4,4],[5,5]]

  for lm in lm_list:

    omegalm0SI_GR, taulm0SI_GR = get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm)

    print 'omega_GR (Hz) values:', omegalm0SI_GR
    print 'freq_GR = omega_GR/2pi (Hz) values', omegalm0SI_GR/(2*pi)
    print 'tau_GR (ms) values:', taulm0SI_GR*1000.

    omegalm0SI_modGR, taulm0SI_modGR = get_sigmalm0SI_modGR(omegalm0SI_GR, taulm0SI_GR, domega220, dtauinv220)

    print 'omega_modGR (Hz) values:', omegalm0SI_modGR
    print 'freq_modGR = omega_modGR/2pi (Hz) values', omegalm0SI_modGR/(2*pi)
    print 'tau_modGR (ms) values:', taulm0SI_modGR*1000.

