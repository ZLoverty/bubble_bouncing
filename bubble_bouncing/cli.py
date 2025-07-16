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
from bubble_bouncing import BounceSimulator
from bubble_bouncing import SimulationParams
from bubble_bouncing import BubbleDataVisualizer
import logging

def run(save_folder, args_dict, exist_ok):
    params = SimulationParams(**args_dict)
    sim = BounceSimulator(save_folder, params, exist_ok=exist_ok)
    logging.basicConfig(
        filename = sim.log_file,
        level = logging.INFO,        # Set the logging level
        format = '%(asctime)s - %(levelname)s - %(message)s',  # Log format
    )
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

def view_traj():
    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default="~/Documents/test", help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()
    vis = BubbleDataVisualizer(args.folder)
    vis.traj_com(mode=args.mode)

def view_morphology():
    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default="~/Documents/test", help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()
    vis = BubbleDataVisualizer(args.folder)
    vis.morphology(mode=args.mode)

def view_oseen():
    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default="~/Documents/test", help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()
    vis = BubbleDataVisualizer(args.folder)
    vis.Oseen_circulation(mode=args.mode)

if __name__=="__main__":
    main()