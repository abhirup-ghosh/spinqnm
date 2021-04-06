import numpy as np
import click
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for status of Gelman-Rubin indices for PE run"
    )
    parser.add_argument(
        "samples_file",
        metavar="samples_file",
        type=str,
        help="path to gelman_rubin.dat file",
    )
    args = parser.parse_args()

    data_loc = args.samples_file
    data = dict(np.genfromtxt(data_loc, dtype=None))

    params_failed = 0

    for param in data.keys():
  	if data[param] < 1.1:
    	    click.echo("{}: ".format(param) + click.style("PASSED", fg="green"))
  	else:
    	    click.echo("{}: ".format(param) + click.style("FAILED", fg="red"))
	    params_failed += 1

    file_name = np.char.replace(args.samples_file, '.dat', '_failed.dat')
    np.savetxt("{}".format(file_name), np.c_[params_failed], fmt='%d')
