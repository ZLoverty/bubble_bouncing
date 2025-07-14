"""
simulate.py
===========
This is the main entry of the bubble bouncing simulation. It is a ready to run script that can take command line arguments to run simulations with different parameters, using the following syntax:

python simulate.py 

"""

from bounce_simulator import BounceSimulator
from pathlib import Path
from utils import parse_params, available_keys
from bubble import SimulationParams
import argparse

parser = argparse.ArgumentParser(f"This script takes in the save_folder as a positional argument. Optional arguments can be passed to set simulation parameters. Available arguments are {available_keys()}. Set using --arg ARG pairs.")
parser.add_argument("--save_folder", type=str, default="~/Documents/test", help="folder to save simulation data.")
parser.add_argument("-f", action="store_true")
# Parse known args; leave unknown ones (like --R, --T, etc.)
args, unknown = parser.parse_known_args()
args_dict = parse_params(unknown)

save_folder = Path(args.save_folder).expanduser().resolve()
params = SimulationParams(**args_dict)
sim = BounceSimulator(save_folder, params, exist_ok=args.f)
sim.run()








