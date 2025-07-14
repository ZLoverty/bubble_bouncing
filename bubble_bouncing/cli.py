"""
cli.py
======
This is the main entry of the bubble bouncing simulation. It is a ready to run script that can take command line arguments to run simulations with different parameters. Use the following syntax:

python cli.py --save_folder FOLDER [-f] [--arg ARG]

or as installed packages:

bcsim --save_folder FOLDER [-f] [--arg ARG]

"""

from pathlib import Path
import argparse
from bubble_bouncing.utils import parse_params, available_keys
from bubble_bouncing.bounce_simulator import BounceSimulator
from bubble_bouncing.bubble import SimulationParams

def run(save_folder, args_dict, exist_ok):
    params = SimulationParams(**args_dict)
    sim = BounceSimulator(save_folder, params, exist_ok=exist_ok)
    sim.run()

def main():
    parser = argparse.ArgumentParser(f"This script takes in the save_folder as a positional argument. Optional arguments can be passed to set simulation parameters. Available arguments are {available_keys()}. Set using --arg ARG pairs.")
    parser.add_argument("--save_folder", type=str, default="~/Documents/test", help="folder to save simulation data.")
    parser.add_argument("-f", action="store_true")
    # Parse known args; leave unknown ones (like --R, --T, etc.)
    args, unknown = parser.parse_known_args()
    save_folder = Path(args.save_folder).expanduser().resolve()
    args_dict = parse_params(unknown)

    run(save_folder, args_dict, args.f)

if __name__=="__main__":
    main()










