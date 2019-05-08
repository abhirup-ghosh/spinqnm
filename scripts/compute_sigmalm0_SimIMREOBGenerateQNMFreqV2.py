import numpy as np
import lal
import lalsimulation as lalsim

def get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm):

  M = m1 + m2

  sigmalm0SI = lal.CreateCOMPLEX16Vector(1)
  lalsim.SimIMREOBGenerateQNMFreqV2(sigmalm0SI, m1, m2, np.array([0., 0., a1z]), np.array([0., 0., a2z]), lm[0], lm[1], 1, lalsim.SEOBNRv4)
  omegalm0SI = np.real(sigmalm0SI.data)
  tauinvlm0SI = np.imag(sigmalm0SI.data)

  return omegalm0SI, tauinvlm0SI

if __name__ == '__main__':

  m1, m2, a1z, a2z = 40, 30, 0, 0
  lm_list = [[2,2],[2,1],[3,3],[4,4],[5,5]]

  for lm in lm_list:

    omegalm0SI, tauinvlm0SI = get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm)

    print '... GR values: omegalm0SI/(2pi): %.2f Hz; taulm0SI: %.2f ms'%(omegalm0SI/(2.*np.pi), 1000./tauinvlm0SI)
