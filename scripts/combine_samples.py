#!/usr/bin/env python3
import argparse
import configparser
import glob
import sys
import subprocess as sp
import os
import numpy as np
import pandas as pd
import h5py
from loguru import logger


def get_git_rev():
    origdir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    git_rev = sp.check_output("git rev-parse HEAD", shell=True).strip().decode()
    try:
        sp.check_call("git diff --quiet", shell=True)
        status = "CLEAN"
    except sp.CalledProcessError:
        status = "DIRTY"
    os.chdir(origdir)
    return f"{git_rev}__{status}"


def grab_webdir(ini_file):
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(ini_file)
    return os.path.join(config.get("paths", "webdir"), "production_postproc")


def get_extra_dead_chains(path):
    """Get the extra chains which may have died"""
    logger.info("Gathering dead or completed chains")
    dead_chains = glob.glob("{}/*.hdf5".format(path))
    chain_list = []
    for chain in dead_chains:
        try:
            fp = h5py.File(chain, "r")
            tmp = fp["/lalinference/lalinference_mcmc/posterior_samples"]
            len_samples = len(tmp)
            if len_samples > 400:
                chain_list.append(chain)
            fp.close()
        except:
            continue
    logger.info(
        "We have collected {} dead/completed chains that have >400 samples".format(
            len(chain_list)
        )
    )
    return chain_list


def sanity_check_chains(combined_chains):
    """Some infinite temp chains occassioanly seem to have a logpost of -np.inf,
    which casuses the downsampling using deltaLogP to fail. Filter out these 
    chains here
    Also check if the length of the chains is below 200 and if it is, reject them.
    This helps guard against very short chains that failed early on and may cause
    cbcBayesMCMC2Pos to fail.
    """
    logger.info("Sanity checking the chains for infinite logpost")
    final_list = []
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
                flag = True
                break
        if not flag:
            final_list.append(chain)
        fp.close()
    logger.info(
        "Out of {} chains we have rejected {} chains due to logpost issues".format(
            len(combined_chains), len(combined_chains) - len(final_list)
        )
    )
    return final_list


def join_command(files, index, destination):
    """Given a list of chains corresponding to a cold chain, construct the combine command"""
    join_root = "/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py37-20200118//bin/cbcBayesCombinePTMCMCh5s"
    full_cmd = "{} --outfile {}/combined_lalinferencemcmc_{}.hdf5 {}".format(
        join_root, destination, index, " ".join(files)
    )
    return full_cmd


def mcmc_to_pos(combined_chains, destination):
    """Combine the posterior files into one, with evidence weighting"""
    mcmc_to_pos_root = "python /home/serguei.ossokine/Sources/lalsuite_for_postproc/lalinference/python/cbcBayesMCMC2pos.py"
    full_cmd = "{} --pos {}/posterior_samples_joint/posterior.hdf5 {} --deltaLogP 7.5 --equal-weighting --combine-only".format(
        mcmc_to_pos_root, destination, " ".join(combined_chains)
    )
    return full_cmd


def cbc_bayes_postproc(outpath, snr_path, posterior_file, psd_files):
    """Generate the old-school postproc pages for insprection"""
    cbc_root = "/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py37-20200118//bin/cbcBayesPostProc --skyres 0.5 --deltaLogP 7.5 --email serguei.ossokine@aei.mpg.de"
    full_cmd = "{} --outpath {} --snr {} {} --psdfiles {}".format(
        cbc_root, outpath, snr_path, posterior_file, ",".join(psd_files)
    )
    return full_cmd


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--event_name", type=str, help="The name of the event to consider")
    args = p.parse_args()
    logger.add(
        "combine_samples_{}.log".format(args.event_name),
        format="{time} {level} {message}",
        level="INFO",
        backtrace=True,
        diagnose=True,
    )
    logger.info("Combining samples together")
    logger.info(
        "This is {} running at revision {}".format(
            os.path.abspath(__file__), get_git_rev()
        )
    )
    # Load the mapping between event_name and actual path to the chains
    # both run here and copied from CIT
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../auxillary_info"))
    mapping = pd.read_json("{}/run_mapping.json".format(root))
    mapping = mapping.sort_values("ID")

    origdir = os.getcwd()
    try:
        row = mapping[mapping["ID"] == args.event_name]
    except:
        logger.error(
            "Could not locate {} in the mapping file. Exiting".format(args.event_name)
        )
        sys.exit(-1)

    local = row.get(key="local").values[0]

    copied = row.get(key="copy").values[0]
    try:
        os.chdir(local)
    except OSError:
        logger.error(
            "Could not go to the local run directory, {}. Exiting".format(local)
        )

    tmpdir = os.path.join(os.getcwd(), "tmp")
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)


    # Grab any useful completed or dead chains
    dead_chains = get_extra_dead_chains(os.path.join(os.getcwd(), "engine"))
    for chain in dead_chains:
        cmd = f"cp {chain}* {tmpdir}"
        sp.call(cmd,shell=True)

    
    chains_local = [os.path.join(tmpdir, "*.hdf5")]
    # If we have chains from other sources
    if copied:
        chains_remote = copied + "/*.hdf5"

    try:
        os.mkdir("{}/combined_samples/".format(tmpdir))
    except OSError:
        pass

    # Deal with local chains
    # Gather all chains, including the hot ones
    offset = 0  # To ensure we number combined chain correctly
    logger.info("Executing cbcBayesCombinePTMCMCh5s on local chains")
    for i, chain_regex in enumerate(chains_local):
        for index, chain in enumerate(sorted(glob.glob(chain_regex))):

            # Get the hot chains
            lst = sorted(glob.glob("{}.[0-9]?".format(chain)))
            # Get the cold chain
            lst.insert(0, chain)
            cmd = join_command(lst, offset, "{}/combined_samples/".format(tmpdir))
            try:
                output = sp.check_output(cmd, shell=True, stderr=sp.STDOUT)
                logger.info(output.decode("utf-8"))
            except sp.CalledProcessError:
                logger.exception("cbcBayesCombinePTMCMCh5s failed!")
                sys.exit(-1)
            offset += 1

    # Deal with copied chains
    if copied:
        logger.info("Executing cbcBayesCombinePTMCMCh5s on remote chains")
        for index, chain in enumerate(sorted(glob.glob(chains_remote))):
            # Get the hot chains
            lst = sorted(glob.glob("{}.[0-9]?".format(chain)))
            # Get the cold chain
            lst.insert(0, chain)
            cmd = join_command(
                lst, offset + index, "{}/combined_samples/".format(tmpdir)
            )
            try:
                output = sp.check_output(cmd, shell=True, stderr=sp.STDOUT)
                logger.info(output.decode("utf-8"))
            except sp.CalledProcessError:
                logger.exception("cbcBayesCombinePTMCMCh5s failed!")
                sys.exit(-1)
    else:
        logger.info("No remote chains found")
    # Now produce posterior from *all* chains. Note that no evidence weighting is *not* done
    combined_chains = glob.glob("{}/combined_samples/combined*".format(tmpdir))
    combined_chains = sanity_check_chains(combined_chains)
    # Create the final output file, posterior.hdf5
    try:
        os.mkdir("{}/posterior_samples_joint".format(tmpdir))
    except OSError:
        pass

    logger.info("Calling cbcBayesMCMC2pos")
    cmd = mcmc_to_pos(combined_chains, "{}".format(tmpdir))
    try:
        output = sp.check_output(cmd, shell=True, stderr=sp.STDOUT)
        logger.info(output.decode("utf-8"))
    except sp.CalledProcessError as e:
        logger.exception("cbcBayesMCMC2pos failed!")
    posterior_file = "{}/posterior_samples_joint/posterior.hdf5".format(tmpdir)
    logger.info(f"Done. The posterior file is in {posterior_file}")

    # Create the old-style postproc pages
    # In case they were not copied, copy the PSD and snr files
    sp.call("cp engine/*PSD* ./tmp/", shell=True)
    sp.call("cp engine/*snr* ./tmp/", shell=True)
    logger.info("Creating old-style postproc pages for inspection")
    outpath = grab_webdir("./config.ini")
    psd_files = [os.path.abspath(x) for x in glob.glob("tmp/*PSD*")]
    try:
        snr_path = os.path.abspath(glob.glob("tmp/*snr*")[0])
    except:
        snr_path = "NA"
    full_cmd = cbc_bayes_postproc(outpath, snr_path, posterior_file, psd_files)
    try:
        output = sp.check_output(full_cmd, shell=True, stderr=sp.STDOUT)
        logger.info(output.decode("utf-8"))
    except sp.CalledProcessError:
        logger.exception("cbcBayesPostProc failed!")
        sys.exit(-1)
    logger.info(f"Done. The webpages can be found at {outpath}")
    os.chdir(origdir)

