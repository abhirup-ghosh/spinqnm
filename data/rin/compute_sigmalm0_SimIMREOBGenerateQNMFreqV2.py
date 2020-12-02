import numpy as np
import lal
import lalsimulation as lalsim

###########################################################
# function to compute GR value of sigmalm0 (in SI units)
# for a single set of m1, m2, a1z, a2z for SEOBNRv4HM
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
# function to compute GR value of sigmalm0 (in SI units)
# for a single set of m1, m2, a1x, a1y, a1z, a2x, a2y, a2z 
# for SEOBNRv4PHM
###########################################################
def get_sigmalm0SI_GR_prec(m1, m2, a1x, a1y, a1z, a2x, a2y, a2z, lm):

  M = m1 + m2

  omegalm0SI = []
  taulm0SI = []

  for idx in range(len(M)):
    sigmalm0SI = lal.CreateCOMPLEX16Vector(1)
    lalsim.SimIMREOBGenerateQNMFreqV2Prec(sigmalm0SI, m1[idx], m2[idx], np.array([a1x[idx], a1y[idx], a1z[idx]]), np.array([a2x[idx], a2y[idx], a2z[idx]]), lm[0], lm[1], 1, lalsim.SEOBNRv4P)

    omegalm0SI.append(np.real((sigmalm0SI.data)[0]))
    taulm0SI.append(1./np.imag((sigmalm0SI.data)[0]))

  return np.asarray(omegalm0SI), np.asarray(taulm0SI)

###########################################################
# function to compute modGR values of sigmalm0 (in SI units)
###########################################################
def get_sigmalm0SI_modGR(omega_GR, tau_GR, domega, dtau):
  return omega_GR*(1. + domega), tau_GR*(1 + dtau)

