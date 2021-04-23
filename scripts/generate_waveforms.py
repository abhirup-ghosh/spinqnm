import matplotlib
matplotlib.use('Agg')

import os

import numpy as np
import lal
import lalsimulation

import compute_sigmalm0_SimIMREOBGenerateQNMFreqV2 as calcqnm
import scipy

import matplotlib.pyplot as plt

# Define binary parameters
nqcCoeffsInput=lal.CreateREAL8Vector(10) ##This will be unused, but it is necessary
m1 = 50.0*lal.MSUN_SI
m2 = 50.0*lal.MSUN_SI
phi_c = 0.
f_start22 = 20. #Frequency of the 22 mode at which the signal starts
distance = 500e6*lal.PC_SI
spin1_z = 0.00346990177835
spin2_z =  0.0440040414498
inclination = 153.95
deltaT = 1./2048.
domega220, dtau220 = 0.1, 0.1
domega210, dtau210 = 0., 0.
domega330, dtau330 = 0., 0.
domega440, dtau440 = 0., 0.
domega550, dtau550 = 0., 0.
alphaNGR = 1.
pNGR = 2.

for idx in range(100):
	alphaNGR = np.random.rand()

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
	lalsimulation.SimInspiralWaveformParamsInsertalphaNGR(paramdict, alphaNGR)
	lalsimulation.SimInspiralWaveformParamsInsertpNGR(paramdict, pNGR)

	hp, hc =  lalsimulation.SimInspiralChooseTDWaveform(m1, m2,  0., 0., spin1_z, 0., 0., spin2_z, distance, inclination, phi_c, 0., 0., 0., deltaT, f_start22, f_start22, paramdict, lalsimulation.SEOBNRv4HM)
#h = hp.data.data - 1j*hc.data.data
#time_array = np.arange(0,len(h)*deltaT,deltaT)

#ampoft = np.abs(h)
#phioft = np.unwrap(np.angle(h))
#Foft = np.gradient(phioft)/np.gradient(time_array)/(2*np.pi)

#plt.figure()
#plt.plot(time_array, hp.data.data)
#plt.savefig('../plots/parspec/waveform_general.png')
