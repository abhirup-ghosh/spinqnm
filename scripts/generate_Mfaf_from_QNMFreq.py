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

######################################################
# Read data
######################################################

# Inspiral
post_loc_insp = '/home/abhirup.ghosh/Documents/Work/O3/2019/May/21/1242442967p4500/G333631/lalinference/20190525_SEOBNRv4_ROM_IMRCT/inspiral_50Hz/1242442967.45-333631/V1H1L1/posterior_samples.dat'
data_insp = np.genfromtxt(post_loc_insp, dtype=None, names=True)
Mf_insp, af_insp = data_insp['mf'],data_insp['af']
Mfaf_insp = np.vstack([Mf_insp, af_insp])

# Ringdown
post_loc_ring = '/home/abhirup.ghosh/Documents/Work/O3/2019/May/21/1242442967p4500/G333631/lalinference/20190525_pSEOBNRv4HM_domega220_dtauinv220/cbcBayes/posterior_samples.dat'
data_ring = np.genfromtxt(post_loc_ring, dtype=None, names=True)
m1_ring, m2_ring, a1z_ring, a2z_ring, domega220_ring, dtau220_ring = data_ring['m1'], data_ring['m2'], data_ring['a1z'], data_ring['a2z'], data_ring['domega220'], data_ring['dtauinv220']
Mf_GR_ring, af_GR_ring = data_ring['mf'],data_ring['af']
Mfaf_GR_ring = np.vstack([Mf_GR_ring, af_GR_ring])

# Compute GR and modGR (omega,tau)
lm = [2,2]
omega_GR_ring, tau_GR_ring = calcqnm.get_sigmalm0SI_GR(m1_ring, m2_ring, a1z_ring, a2z_ring, lm)
omega_modGR_ring, tau_modGR_ring = calcqnm.get_sigmalm0SI_modGR(omega_GR_ring, tau_GR_ring, domega220_ring, dtau220_ring)
Mf_modGR_ring, af_modGR_ring = Mjfinal220(omega_modGR_ring, tau_modGR_ring)

idx, = (np.where((af_modGR_ring > 0) & (af_modGR_ring < 1)))

Mfaf_modGR_ring = np.vstack([Mf_modGR_ring[idx], af_modGR_ring[idx]])

fig1 = corner.corner(np.transpose(Mfaf_modGR_ring), labels=[r'$M$',r'$j$'],
                    smooth=True,color='blue',plot_contours=True,levels=([0.9]),
                    range=([0,600],[0,1]),plot_datapoints=False)

fig2 = corner.corner(np.transpose(Mfaf_GR_ring), labels=[r'$M$',r'$j$'],
                    smooth=True,color='green',plot_contours=True,levels=([0.9]),
                    range=([0,600],[0,1]),plot_datapoints=False, fig=fig1)

corner.corner(np.transpose(Mfaf_insp), labels=[r'$M$',r'$j$'],
                    smooth=True,color='black',plot_contours=True,levels=([0.9]),
                    range=([0,600],[0,1]),plot_datapoints=False, fig=fig2)

plt.minorticks_on()
plt.savefig('./IMRCT_S190521g.png')
