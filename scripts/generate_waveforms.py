# source ~/opt/lalsuite_padynamics_RD_parspec_e6a04843_20210713/etc/lalsuiterc

import matplotlib
matplotlib.use('Agg')

import os

import numpy as np
import lal
import lalsimulation

import compute_sigmalm0_SimIMREOBGenerateQNMFreqV2 as calcqnm
import scipy

import matplotlib.pyplot as plt

def generate_waveform(domega220, dtau220, alphaNGR, pNGR):

    paramdict = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertDOmega220(paramdict, domega220)
    lalsimulation.SimInspiralWaveformParamsInsertDTau220(paramdict, dtau220)
    lalsimulation.SimInspiralWaveformParamsInsertDOmega210(paramdict, domega210)
    lalsimulation.SimInspiralWaveformParamsInsertDTau210(paramdict, dtau210)
    lalsimulation.SimInspiralWaveformParamsInsertDOmega330(paramdict, domega330)
    lalsimulation.SimInspiralWaveformParamsInsertDTau330(paramdict, dtau330)
    lalsimulation.SimInspiralWaveformParamsInsertDOmega440(paramdict, domega440)
    lalsimulation.SimInspiralWaveformParamsInsertDTau440(paramdict, dtau440)
    lalsimulation.SimInspiralWaveformParamsInsertDOmega550(paramdict, domega550)
    lalsimulation.SimInspiralWaveformParamsInsertDTau550(paramdict, dtau550)
    lalsimulation.SimInspiralWaveformParamsInsertAlphaNGR(paramdict, alphaNGR)
    lalsimulation.SimInspiralWaveformParamsInsertPNGR(paramdict, pNGR)

    hp, hc =  lalsimulation.SimInspiralChooseTDWaveform(m1, m2,  0., 0., spin1_z, 0., 0., spin2_z, distance, inclination, phi_c, 0., 0., 0., deltaT, f_start22, f_start22, paramdict, lalsimulation.SEOBNRv4HM)
    time_array = np.arange(0,len(hp.data.data)*deltaT,deltaT)

    return time_array, hp.data.data, hc.data.data

# Define binary parameters
nqcCoeffsInput=lal.CreateREAL8Vector(10) ##This will be unused, but it is necessary
m1 = 35.0*lal.MSUN_SI
m2 = 35.0*lal.MSUN_SI
phi_c = 0.
f_start22 = 20. #Frequency of the 22 mode at which the signal starts
distance = 500e6*lal.PC_SI
spin1_z = 0.00346990177835
spin2_z =  0.0440040414498
inclination = 153.95
deltaT = 1./2048.
domega210, dtau210 = 0., 0.
domega330, dtau330 = 0., 0.
domega440, dtau440 = 0., 0.
domega550, dtau550 = 0., 0.

pNGR = 2.
alphaNGR = 100.

z = distance * 70000.0 / 1.0e6 / lal.PC_SI / lal.C_SI
gamma = (alphaNGR * lal.C_SI * lal.C_SI * (1. + z)/ lal.G_SI/ (m1+m2))**pNGR

plt.figure()

# plot pSEOB data

t, hp, hc = np.loadtxt("../data/parspec_testing/pSEOB.dat", unpack=True)
plt.plot(t, hp, color='r')

# plot data with no deviation
t, hp, hc = generate_waveform(0, 0, 100, pNGR)
plt.plot(t, hp, color='g', ls='dashdot')
"""
for idx in range(10000):

	domega220 =  np.random.uniform(0.,1.)
	dtau220 =  np.random.uniform(0.,1.)

	t, hp, hc = generate_waveform(domega220, dtau220, 100, pNGR)
	#t, hp, hc = generate_waveform(0.394314, -0.888588, 100, pNGR)
	plt.plot(t, hp, alpha=0.1, color='k', lw=0.1)

plt.xlim([0.32, 0.38])
plt.savefig('../plots/parspec/pSEOB_parspec-no-dev_comparison_p2_df0_dtau0.png')
"""
