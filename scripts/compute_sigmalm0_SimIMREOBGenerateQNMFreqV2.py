import numpy as np
import lal
import lalsimulation as lalsim

###########################################################
# function to compute GR value of sigmalm0 (in SI units)
# for a single set of m1, m2, a1z, a2z
###########################################################
def get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm):

  M = m1 + m2

  sigmalm0SI = lal.CreateCOMPLEX16Vector(1)
  lalsim.SimIMREOBGenerateQNMFreqV2(sigmalm0SI, m1, m2, np.array([0., 0., a1z]), np.array([0., 0., a2z]), lm[0], lm[1], 1, lalsim.SEOBNRv4)
  omegalm0SI = np.real(sigmalm0SI.data)/(2.*np.pi)
  taulm0SI = 1./np.imag(sigmalm0SI.data)

  return omegalm0SI, taulm0SI


###########################################################
# function to compute GR values of sigmalm0 (in SI units)
# for m1, m2, a1z, a2z arrays; returns a numpy array
###########################################################
def get_sigmalm0SI_GR_list(m1, m2, a1z, a2z, lm):

  M = m1 + m2

  return np.asarray([get_sigmalm0SI_GR(m1[idx], m2[idx], a1z[idx], a2z[idx], lm)[0][0] for idx in range(len(m1))]), np.asarray([get_sigmalm0SI_GR(m1[idx], m2[idx], a1z[idx], a2z[idx], lm)[1][0] for idx in range(len(m1))])

###########################################################
# function to compute modGR values of sigmalm0 (in SI units)
###########################################################
def get_sigmalm0SI_modGR_list(omega_GR_list, tau_GR_list, domega_list, dtau_list):
  return omega_GR_list*(1. + domega_list), tau_GR_list*(1 + dtau_list)

if __name__ == '__main__':

  m1, m2, a1z, a2z = 40, 30, 0, 0
  lm_list = [[2,2],[2,1],[3,3],[4,4],[5,5]]

  for lm in lm_list:

    omegalm0SI, taulm0SI = get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm)

    print '... GR values: omegalm0SI: %.2f Hz; taulm0SI: %.2f ms'%(omegalm0SI, taulm0SI*1000.)

  m1, m2, a1z, a2z = np.random.randn(100) + 40, np.random.randn(100)+30, np.zeros(100), np.zeros(100)
  lm = [2,2]
  omegalm0SI_list, taulm0SI_list = get_sigmalm0SI_GR_list(m1, m2, a1z, a2z, lm)
  print omegalm0SI_list, taulm0SI_list
