import numpy as np
import scipy
from scipy import interpolate

abyM,Momega_r,Momega_i,alm_r,alm_i = np.genfromtxt('../src/l2/n1l2m2.dat', unpack=True)

abyM_vs_Momega_r_times_Momega_i_interp_obj = scipy.interpolate.interp1d(Momega_r*Momega_i, abyM)
Momega_r_vs_abyM_interp_obj = scipy.interpolate.interp1d(abyM, Momega_r)


