import numpy as np
import glob
import h5py
from optparse import OptionParser

def sanity_check_chains(combined_chains):
    """Some infinite temp chains occassioanly seem to have a logpost of -np.inf,
    which casuses the downsampling using deltaLogP to fail. Filter out these
    chains here
    Also check if the length of the chains is below 200 and if it is, reject them.
    This helps guard against very short chains that failed early on and may cause
    cbcBayesMCMC2Pos to fail.
    """
    #logger.info("Sanity checking the chains for infinite logpost")
    final_list = []
    rejected_list = []
    for chain in combined_chains:
        flag = False
        fp = h5py.File(chain, "r")
        tmp = fp["/lalinference/lalinference_mcmc"]
        cold_samples = fp["/lalinference/lalinference_mcmc/posterior_samples"][()]
        if len(cold_samples) < 200:
            continue
        for i in range(1, 8):
            data = fp["/lalinference/lalinference_mcmc/chain_{:02d}".format(i)]
            logpost = data["logpost"]
            if (len(np.where(logpost > -1e8)[0]) / len(logpost)) < 0.99:
		rejected_list.append(chain)
                flag = True
                break
        if not flag:
            final_list.append(chain)
        fp.close()
    #logger.info(
    #    "Out of {} chains we have rejected {} chains due to logpost issues".format(
    #        len(combined_chains), len(combined_chains) - len(final_list)
    #    )
    #)
    return final_list, rejected_list


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-p", "--post-loc", dest="post_loc", help="path to directory containing the cbcBayesPostProc posterior_samples.dat output")
    (options, args) = parser.parse_args()
    post_loc = options.post_loc

    combined_chains = glob.glob(post_loc + "/combine*")
    final_list, rejected_list = sanity_check_chains(combined_chains)

    print (' '.join(final_list))
