import numpy as np
import h5py

import sys
sys.path.append("../../scripts")

import compute_sigmalm0_SimIMREOBGenerateQNMFreqV2 as calcqnm

data_loc_full_pe_dict = {"S190519bj":"../../data/rin/S190519bj/rin_S190519bj_pseobnrv4hm.h5", 
              "S190521r":"../../data/rin/S190521r/rin_S190521r_pseobnrv4hm.h5",
              "S190630ag":"../../runs/O3a/S190630ag/220/cbcBayes/posterior_samples.dat",
              "S190828j":"../../runs/O3a/S190828j/220/cbcBayes/posterior_samples.dat",
              "S190910s":"../../data/rin/S190910s/rin_S190910s_pseobnrv4hm.h5",
              "GW150914":"../../runs/O1O2/GW150914/220/cbcBayes/posterior_samples.dat",
              "GW170104":"../../runs/O1O2/GW170104/220/cbcBayes/posterior_samples.dat",
              "GW170729":"../../runs/O1O2/GW170729/220/cbcBayes/posterior_samples.dat"}

for event in data_loc_full_pe_dict.keys():

	if data_loc_full_pe_dict[event].endswith('.h5'):

		pos_file = h5py.File(data_loc_full_pe_dict[event], "r")
		samples = pos_file["pSEOBNRv4HM"]["posterior_samples"]

		m1, m2, a1z, a2z, domega, dtau = samples["mass_1"], samples["mass_2"],\
                                         samples["spin_1z"], samples["spin_2z"],\
                                         samples["domega220"], samples["dtau220"]

	else:
		samples = np.genfromtxt(data_loc_full_pe_dict[event], dtype=None, names=True)
		m1, m2, a1z, a2z, domega, dtau = samples["m1"], samples["m2"],\
                                         samples["a1z"], samples["a2z"],\
                                         samples["domega220"], samples["dtau220"]
    
	omega_GR, tau_GR = calcqnm.get_sigmalm0SI_GR(m1, m2, a1z, a2z, [2,2])
	freq_GR = omega_GR/(2.*np.pi)
        
	omega_modGR, tau_modGR = calcqnm.get_sigmalm0SI_modGR(omega_GR, tau_GR, domega, dtau)
	freq_modGR = omega_modGR/(2.*np.pi)    

	print(f"{event}: freq_modGR = [{min(freq_modGR):.2f}, {max(freq_modGR):.2f}], tau_modGR = [{min(tau_modGR):.4f}, {max(tau_modGR):.4f}]")
    
	np.savetxt(f"../../data/rin/{event}/rin_{event}_pseobnrv4hm_domega_220.dat.gz",np.c_[domega])
	np.savetxt(f"../../data/rin/{event}/rin_{event}_pseobnrv4hm_dtau_220.dat.gz",np.c_[dtau])
	np.savetxt(f"../../data/rin/{event}/rin_{event}_pseobnrv4hm_freq_220_modGR.dat.gz",np.c_[freq_modGR])
	np.savetxt(f"../../data/rin/{event}/rin_{event}_pseobnrv4hm_tau_220_modGR.dat.gz",np.c_[tau_modGR])
