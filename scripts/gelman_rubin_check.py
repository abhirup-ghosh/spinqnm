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

    for param in data.keys():
  	if data[param] < 1.1:
    	    click.echo("{}: ".format(param) + click.style("PASSED", fg="green"))
  	else:
    	    click.echo("{}: ".format(param) + click.style("FAILED", fg="red"))
