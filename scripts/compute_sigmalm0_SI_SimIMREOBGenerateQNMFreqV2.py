import numpy as np
import lal
import lalsimulation as lalsim

def get_sigmalm0_SI(m1, m2, chi1z, chi2z, lm):

  M = m1 + m2

  sigmalm0_SI = lal.CreateCOMPLEX16Vector(1)
  lalsim.SimIMREOBGenerateQNMFreqV2(sigmalm0_SI, m1, m2, np.array([0., 0., chi1z]), np.array([0., 0., chi2z]), lm[0], lm[1], 1, lalsim.SEOBNRv4)
  omegalm0_SI = np.real(sigmalm0_SI.data)
  tauinvlm0_SI = np.imag(sigmalm0_SI.data)

  return omegalm0_SI, tauinvlm0_SI

if __name__ == '__main__':

  m1, m2, chi1z, chi2z = 40, 30, 0, 0
  lm_list = [[2,2],[2,1],[3,3],[4,4],[5,5]]

  for lm in lm_list:

    omegalm0_SI, tauinvlm0_SI = get_sigmalm0_SI(m1, m2, chi1z, chi2z, lm)
    print omegalm0_SI/(2.*np.pi), tauinvlm0_SI
