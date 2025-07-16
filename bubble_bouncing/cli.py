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

test_folder = "~/Documents/.bcsim_test"

def main():
    parser = argparse.ArgumentParser(f"This script takes in the save_folder as a positional argument. Optional arguments can be passed to set simulation parameters. Available arguments are {available_keys()}. Set using --arg ARG pairs.")
    parser.add_argument("--save_folder", type=str, default=test_folder, help="folder to save simulation data.")
    parser.add_argument("-f", action="store_true")
    parser.add_argument('--log-level', '-l',
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    # Parse known args; leave unknown ones (like --R, --T, etc.)
    args, unknown = parser.parse_known_args()
    save_folder = Path(args.save_folder).expanduser().resolve()
    args_dict = parse_params(unknown)
    params = SimulationParams(**args_dict)
    log_level_str = args.log_level
    sim = BounceSimulator(save_folder, params, exist_ok=args.f)
    
    
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")

    logging.basicConfig(
        filename = sim.log_file,
        level = numeric_level,        # Set the logging level
        format = '%(asctime)s - %(levelname)s - %(message)s',  # Log format
    )
    sim.run()
    

def view_traj():
    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default=test_folder, help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()
    vis = BubbleDataVisualizer(args.folder)
    vis.traj_com(mode=args.mode)

def view_morphology():
    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default=test_folder, help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()
    vis = BubbleDataVisualizer(args.folder)
    vis.morphology(mode=args.mode)

def view_oseen():
    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default=test_folder, help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()
    vis = BubbleDataVisualizer(args.folder)
    vis.Oseen_circulation(mode=args.mode)

if __name__=="__main__":
    main()