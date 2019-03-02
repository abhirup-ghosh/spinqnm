import lalsimulation,lal
import numpy as np
import subprocess
import os, sys
from numpy import sqrt, sin, cos, pi
import matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import emcee
from pycbc  import  detector
from pycbc import psd
import corner
from optparse import OptionParser
import time
import standard_gwtransf as gw
import scipy
from scipy import interpolate
from scipy.signal import argrelextrema

""" taper time domain data. h is a numpy array (Ref. Eq. (3.35) of gr-qc/0001023) """
def taper_waveform(h):
        h_temp = h
        peakind = np.array(argrelextrema(abs(h_temp), np.greater)).flatten()
        idx_peak2 = peakind[1]          # index of second extremum
        startind = np.flatnonzero(h_temp)[0]            # index of first non-zero data point

        # taper from start to second extremum 
        n = idx_peak2 - startind
        # do the taper using formula Eq. (3.35) of gr-qc/0001023.
        h_temp[startind] = 0
        for i in range(startind+1, startind+n-2):
                z = (n - 1.)/(i-startind) + (n-1.)/(i-startind - (n-1.))
                h_temp[i] = h_temp[i]*1./(np.exp(z) + 1)
        return h_temp


def seobnrv4hm_wfm(deltaT, Mc, q, f_start22, distance, spin1_z, spin2_z, phi_c, domega220, dtauinv220, domega210, dtauinv210, domega330, dtauinv330, domega440, dtauinv440, domega550, dtauinv550, nqcCoeffsInput, inclination):

	m1, m2 = gw.comp_from_mcq(Mc, q)
	print domega220, dtauinv220, domega210, dtauinv210, domega330, dtauinv330, domega440, dtauinv440, domega550, dtauinv550

	sphtseries, dyn, dynHi = lalsimulation.SimIMRSpinAlignedEOBModes(deltaT, m1, m2, f_start22, distance, spin1_z, spin2_z,41, 0., 0., 0.,0.,0.,0.,0.,0.,1.,1., domega220, dtauinv220, domega210, dtauinv210, domega330, dtauinv330, domega440, dtauinv440, domega550, dtauinv550, nqcCoeffsInput, 0)

        # Read the modes
        hlm = {}

        ##55 mode
        modeL = sphtseries.l
        modeM = sphtseries.m
        print 'Loading mode', modeL, modeM
        h55 = sphtseries.mode.data.data #This is h_55
        hlm[(modeL, modeM)] = h55

        ##44 mode
        modeL = sphtseries.next.l
        modeM = sphtseries.next.m
        print 'Loading mode', modeL, modeM
        h44 = sphtseries.next.mode.data.data #This is h_44
        hlm[(modeL, modeM)] = h44

        ##21 mode
        modeL = sphtseries.next.next.l
        modeM = sphtseries.next.next.m
        print 'Loading mode', modeL, modeM
        h21 = sphtseries.next.next.mode.data.data #This is h_21
        hlm[(modeL, modeM)] = h21

        ##33 mode
        modeL = sphtseries.next.next.next.l
        modeM = sphtseries.next.next.next.m
        print 'Loading mode', modeL, modeM
        h33 = sphtseries.next.next.next.mode.data.data #This is h_33
        hlm[(modeL, modeM)] = h33
        
        ##22 mode
        modeL = sphtseries.next.next.next.next.l
        modeM = sphtseries.next.next.next.next.m
        print 'Loading mode', modeL, modeM
        h22 = sphtseries.next.next.next.next.mode.data.data #This is h_22
        hlm[(modeL, modeM)] = h22

        ##time array (s)
        time_array = np.arange(0,len(h22)*deltaT,deltaT)

        # Generate the full waveform
        paramdict = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertDOmega220(paramdict, domega220)
        lalsimulation.SimInspiralWaveformParamsInsertDTauInv220(paramdict, dtauinv220)
        lalsimulation.SimInspiralWaveformParamsInsertDOmega210(paramdict, domega210)
        lalsimulation.SimInspiralWaveformParamsInsertDTauInv210(paramdict, dtauinv210)
        lalsimulation.SimInspiralWaveformParamsInsertDOmega330(paramdict, domega330)
        lalsimulation.SimInspiralWaveformParamsInsertDTauInv330(paramdict, dtauinv330)
        lalsimulation.SimInspiralWaveformParamsInsertDOmega440(paramdict, domega440)
        lalsimulation.SimInspiralWaveformParamsInsertDTauInv440(paramdict, dtauinv440)
        lalsimulation.SimInspiralWaveformParamsInsertDOmega550(paramdict, domega550)
        lalsimulation.SimInspiralWaveformParamsInsertDTauInv550(paramdict, dtauinv550)

        hp, hc =  lalsimulation.SimInspiralChooseTDWaveform(m1, m2,  0., 0., spin1_z, 0., 0., spin2_z, distance, inclination, phi_c, 0., 0., 0., deltaT, f_start22, f_start22, paramdict, lalsimulation.SEOBNRv4HM)

        ##time array (s)
        time_array = np.arange(0,len(h22)*deltaT,deltaT)

    	return time_array, hp.data.data, hc.data.data

def lnlike(param_vec, data, freq, psd, f_low, f_cut):
        """
        compute the log likelihood
        
        inputs: 
        param_vec : vector of parameters 
        dr, di, 
        freq : Fourier freq 
        psd : psd vector 
        flow,fcut
        
        output: 
        log_likelhood 
        """
        df = np.diff(freq)[0]

        N_low=np.int((f_low-freq[0])/df)
        N_cut=np.int((f_cut-freq[0])/df)

        Nls=np.int(f_low/df)  #N_low_signal
        Ncs=np.int(f_cut/df)  #N_cut_signal

        # unpacking the parameter vector 
        domega220, = param_vec

	t, hp, hc = seobnrv4hm_wfm(deltaT, Mc, q, f_start22, distance, spin1_z, spin2_z, phi_c, domega220, dtauinv220, domega210, dtauinv210, domega330, dtauinv330, domega440, dtauinv440, domega550, dtauinv550, nqcCoeffsInput, inclination)

        # compute antenna patterns 
        Fp,Fc = detector.overhead_antenna_pattern(ra, np.arcsin(sin_dec), pol)

        signal=Fp*hp+Fc*hc

	N = len(signal)
        f = np.fft.fftfreq(N, d=deltaT)
        #signal_freq = np.fft.fft(taper_waveform(signal))*deltaT
        signal_freq = np.fft.fft(signal)*deltaT

        like = -2.*df*np.real(np.dot(data[N_low:N_cut]-signal[Nls:Ncs],np.conj((data[N_low:N_cut]-signal[Nls:Ncs])/psd[N_low:N_cut])))

        return like#log-likelihood

def lnprior(param_vec):
        domega220 = param_vec
        if -1. < domega220 < 1.:
        	return 0
	return -np.inf



def lnprob(param_vec):
        lp = lnprior(param_vec)
        if not np.isfinite(lp):
                return -np.inf
        return lp + lnlike(param_vec, data, freq, psd, f_low, f_cut)


##########################################################
###################### MAIN ##############################
##########################################################

nqcCoeffsInput=lal.CreateREAL8Vector(10) ##This will be unused, but it is necessary
m1 = 50.0*lal.MSUN_SI
m2 = 50.0*lal.MSUN_SI
Mc, q = gw.mcq_from_comp(m1,m2)
phi_c = 0.
f_start22 = 8. #Frequency of the 22 mode at which the signal starts
distance = 500.0*lal.PC_SI
spin1_z = 0.
spin2_z =  0.
deltaT = 1./16384.
inclination = 0.
ra, sin_dec, pol = 0., 0., 0.
domega220_init = 0.
dtauinv220 = 0.
domega210 = 0.
dtauinv210 = 0.
domega330 = 0.
dtauinv330 = 0.
domega440 = 0.
dtauinv440 = 0.
domega550 = 0.
dtauinv550 = 0.

f_low = 20.
f_cut = 999.

t, hp, hc = seobnrv4hm_wfm(deltaT, Mc, q, f_start22, distance, spin1_z, spin2_z, phi_c, domega220_init, dtauinv220, domega210, dtauinv210, domega330, dtauinv330, domega440, dtauinv440, domega550, dtauinv550, nqcCoeffsInput, inclination)
Fp,Fc = detector.overhead_antenna_pattern(ra, np.arcsin(sin_dec), pol)
signal=Fp*hp+Fc*hc
N = len(signal)
freq = np.fft.fftfreq(N, d=deltaT)
df = np.diff(freq)[0]
#signal_freq = np.fft.fft(taper_waveform(signal))*deltaT
data = np.fft.fft(signal)*deltaT
psd = psd.aLIGOZeroDetHighPower(len(data), df, f_low)
psd = np.asarray(psd)

ndim, nwalkers = 1, 100
num_threads = 30
num_iter = 100

# create initial walkers
result = domega220_init

pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=num_threads)
sampler.run_mcmc(pos, num_iter)

domega220_chain = sampler.chain[:, :, 0].T
samples = sampler.chain[:, :, :].reshape((-1, ndim))

plt.figure(figsize=(15,5))
plt.subplot(111)
plt.plot(domega220_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(domega220_init + np.std(domega220_chain, axis=1), 'r')
plt.axhline(y=domega220_init, color='g')
plt.ylabel('domega220')
plt.savefig('./domega220_chain.png')


