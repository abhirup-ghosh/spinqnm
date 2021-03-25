import numpy as np

data_dict = {
	'GW150914':'/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O1O2/GW150914/GR/cbcBayes/posterior_samples.dat',
	'GW170104':'/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O1O2/GW170104/GR/cbcBayes/posterior_samples.dat',
	'GW190519a': '/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O3a/GR_runs_posterior_samples/GW190519a_posterior_samples.dat',
	'GW190521b': '/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O3a/GR_runs_posterior_samples/GW190521b_posterior_samples.dat',
	'S190630ag':'/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O3a/S190630ag/GR/cbcBayes/posterior_samples.dat',
	'S190828j':'/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O3a/S190828j/GR/cbcBayes/posterior_samples.dat',
	'GW190910a': '/home/abhirup.ghosh/Documents/Work/spinqnm/runs/O3a/GR_runs_posterior_samples/GW190910a_posterior_samples.dat'
}

for event in data_dict.keys():
	data = np.genfromtxt(data_dict[event], dtype=None, names=True)

	print(event)
	for param in ['mf_evol', 'af_evol']:
	
		xf = data[param]
		xf_med = np.median(xf)

		xf_lower = xf_med - np.percentile(xf, 5)
		xf_upper = np.percentile(xf, 95) - xf_med

		if param == 'mf_evol':
			print("$%.1f^{+%0.1f}_{-%0.1f}$ &"%(xf_med, xf_upper, xf_lower))
		else:
			print("$%.2f^{+%0.2f}_{-%0.2f}$"%(xf_med, xf_upper, xf_lower))

	
