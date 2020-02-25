#!/usr/bin/env python
import glob
import os
import subprocess
import shutil
import argparse
import tqdm

if __name__ =="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_name",type=str,help="The batch name of the run to postprocess")
    args = p.parse_args()
    try:
        os.mkdir("postprocess")
    except:
        print("Directory postprocess exists, not overwriting")
        exit(-1)

    cmd = """condor_q -constraint 'JobBatchName == "{}"' -af ClusterId""".format(args.batch_name)
    joblist = [x for x in subprocess.check_output(cmd,shell=True).split("\n") if x]
    #joblist = [str(x, encoding='utf-8') for x in subprocess.check_output(cmd,shell=True).split(b"\n") if x]
    print("Found {} jobs running for this batch name".format(len(joblist)))
    postprocessdir = os.getcwd()+"/postprocess"
    
    for job in tqdm.tqdm(joblist):
        #print(job)
        cmd = """condor_ssh_to_job {}  "cp *.hdf5* {}" """.format(job,postprocessdir)
        try:
            subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
        except:
            pass
