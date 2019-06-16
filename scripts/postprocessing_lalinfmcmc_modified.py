import numpy as np
import glob
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-p", "--post-loc-root", dest="post_loc_root", help="path to post loc")
(options, args) = parser.parse_args()

post_loc_root = options.post_loc_root

psd_files = glob.glob(post_loc_root + '/engine/*PSD.dat')
psd_files_arg = ','.join(psd_files)

snr_file = glob.glob(post_loc_root + '/engine/*snr.txt')

lalinfmcmc_files = glob.glob(post_loc_root + '/engine/lalinferencemcmc*.hdf5')
lalinfmcmc_files_arg = ' '.join(lalinfmcmc_files)

out_dir = post_loc_root + '/cbcBayes'
os.system('mkdir -p %s'%out_dir)

command = 'cbcBayesPostProc --snr %s --skyres 0.5 --psdfiles %s --deltaLogP 6 --outpath %s %s'%(snr_file, psd_files_arg, out_dir, lalinfmcmc_files_arg)
print command
os.system(command)
