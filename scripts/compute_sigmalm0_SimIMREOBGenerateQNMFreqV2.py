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

###########################################################
# function to compute modGR values of sigmalm0 (in SI units)
###########################################################
def get_sigmalm0SI_modGR_parspec(m1, m2, a1z, a2z, domega220, dtau220):

    f_220 = np.zeros(len(m1))
    tau_220 = np.zeros(len(m1))
    
    for idx in range(len(m1)):
    
        i, finalMass, finalSpin = lalsim.SimIMREOBFinalMassSpin(m1[idx], m2[idx], \
                                                             np.array([0., 0., a1z[idx]]), np.array([0., 0., a2z[idx]]), \
                                                             lalsim.SEOBNRv4)
        
        
        omegaNGR = 0.3737 * (1. + domega220[idx]);
        omegaGR = 0.1258*finalSpin + 0.0717*pow(finalSpin,2) + 0.0480*pow(finalSpin,3) + 0.0350*pow(finalSpin,4);

        tauNGR = 11.2407 * (1. + dtau220[idx]);
        tauGR = 0.2522*finalSpin + 0.6649*pow(finalSpin,2) + 0.5866*pow(finalSpin,3) + 0.5797*pow(finalSpin,4);
        
        omega_220 = 1./finalMass/(m1[idx]+m2[idx])/lal.MTSUN_SI*(omegaNGR + omegaGR)
        f_220[idx] = omega_220/(2.*np.pi)
        
        tau_220[idx] = finalMass * (m1[idx]+m2[idx]) * lal.MTSUN_SI * (tauGR + tauNGR)
        
    return f_220, tau_220

if __name__ == '__main__':

  post_loc = '/home/abhirup.ghosh/Documents/Work/O3/2019/May/21/1242442967p4500/G333631/lalinference/20190525_pSEOBNRv4HM_domega220_dtau220/cbcBayes/posterior_samples.dat'
  data = np.genfromtxt(post_loc, names=True, dtype=None)
  m1, m2, a1z, a2z, domega220, dtau220 = data['m1'], data['m2'], data['a1z'], data['a2z'], data['domega220'], data['dtau220']

  lm_list = [[2,2],[2,1],[3,3],[4,4],[5,5]]

  for lm in lm_list:

    omegalm0SI_GR, taulm0SI_GR = get_sigmalm0SI_GR(m1, m2, a1z, a2z, lm)

    print('omega_GR (Hz) values:', omegalm0SI_GR)
    print('freq_GR = omega_GR/2pi (Hz) values', omegalm0SI_GR/(2*pi))
    print('tau_GR (ms) values:', taulm0SI_GR*1000.)

    omegalm0SI_modGR, taulm0SI_modGR = get_sigmalm0SI_modGR(omegalm0SI_GR, taulm0SI_GR, domega220, dtau220)

    print('omega_modGR (Hz) values:', omegalm0SI_modGR)
    print('freq_modGR = omega_modGR/2pi (Hz) values', omegalm0SI_modGR/(2*pi))
    print('tau_modGR (ms) values:', taulm0SI_modGR*1000.)

