import numpy as np
import h5py

post_loc = '/home/abhirup.ghosh/Documents/Work/spinqnm/runs/20190412_GW170729_pSEOBNRv4HM_tgr_4baa4caf_domega220_dtauinv220_H1L1_nyq_patch_dtauinv220_m0p99_5p0/engine/lalinferencemcmc-0-H1L1-1185389807.33-96.hdf5'

f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
