import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import corner

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import compute_sigmalm0_SimIMREOBGenerateQNMFreqV2 as calcqnm

def Mjfinal220(omega220,tau220):
    
    jf=1-pow((omega220*tau220/2.-0.7)/(1.4187),-1/0.4990)
    Mf=((1.5251 - 1.1568*pow((1 - jf),0.1292))/omega220)*(cc*cc*cc/G/Msun)

    return Mf,jf


Msun = 1.9885469549614615*10**30
G = 6.67408*10**(-11)
cc = 299792458

post_loc_insp = '/home/abhirup.ghosh/Documents/Work/O3/2019/May/21/1242459857p4634/G333674/lalinference/20190525_SEOBNRv4_ROM_IMRCT/inspiral/1242459857.46-333674/H1L1/posterior_samples.dat'
data_insp = np.genfromtxt(post_loc_insp, dtype=None, names=True)

post_loc_ring = '/home/abhirup.ghosh/Documents/Work/O3/2019/May/21/1242459857p4634/G333674/lalinference/20190525_pSEOBNRv4HM_domega220_dtauinv220_nonROQ/cbcBayes/posterior_samples.dat'
data_ring = np.genfromtxt(post_loc_ring, dtype=None, names=True)

Mf_insp, af_insp = data_insp['mf'],data_insp['af']
print Mf_insp, af_insp

#MJ_insp = np.vstack([data_insp['mf'],data_insp['af']])

m1_ring, m2_ring, a1z_ring, a2z_ring, domega220_ring, dtau220_ring = data_ring['m1'], data_ring['m2'], data_ring['a1z'], data_ring['a2z'], data_ring['domega220'], data_ring['dtauinv220']
lm = [2,2]
omega_GR_ring, tau_GR_ring = calcqnm.get_sigmalm0SI_GR(m1_ring, m2_ring, a1z_ring, a2z_ring, lm)
omega_modGR_ring, tau_modGR_ring = calcqnm.get_sigmalm0SI_modGR(omega_GR_ring, tau_GR_ring, domega220_ring, dtau220_ring)

print omega_modGR_ring, tau_modGR_ring

#MJ = Mjfinal220(omega_modGR_ring, tau_modGR_ring)
#MJ_ring = np.vstack([MJ[0][(MJ[1]>0) & (MJ[1]<1)], MJ[1][(MJ[1]>0) & (MJ[1]<1)]])

Mf_ring, af_ring = Mjfinal220(omega_modGR_ring, tau_modGR_ring)

plt.figure(figsize=(5,5))
plt.scatter(Mf_insp, af_insp, 'k')
plt.scatter(Mf_ring, af_ring, 'r')
plt.savefig('./IMRCT_S190521r.png')
