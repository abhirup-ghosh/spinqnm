# Copyright 2021, Abhirup Ghosh <abhirup.ghosh.184098@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

event_list = ["GW150914", "GW190521" ]
run_list = ["0p0_GR", "0p0", "0p1_GR", "0p1", "0p5_GR", "0p5"]
label_list = [np.char.replace(label,"_",":") for label in run_list]

plt.figure(figsize=(10,5))

for event in event_list:
  plt.figure(figsize=(10,5))
  for (label,run) in zip(label_list,run_list):
    data = np.genfromtxt(f"/Users/abhirupghosh/Documents/Work/spinqnm/runs/nonGR/{event}-like/domega220_dtau220_{run}_widerdist/cbcBayes/posterior_samples.dat", dtype=None, names=True)
    
    plt.hist(data["matched_filter_snr"], histtype="step", bins=50, density=True, label=f"{event}:{label}")

  #plt.xlim(left=15)
  plt.xlabel("$\\rho_{SNR}$")
  plt.ylabel("P($\\rho_{SNR}$)")
  plt.legend(loc="best")
  plt.savefig(f"./compare_snr_nonGR_runs_{event}.png")
    
