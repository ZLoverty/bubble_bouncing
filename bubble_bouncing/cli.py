"""
cli.py
======
This is the main entry of the bubble bouncing simulation. It is a ready to run script that can take command line arguments to run simulations with different parameters. Install the package either directly from the GitHub repo:

```
pip install git+https://github.com/ZLoverty/bubble_bouncing.git
```

or, clone the git repo and install as editable package for development

```
git clone https://github.com/ZLoverty/bubble_bouncing.git
conda env create -f environment.yaml
conda activate bcsim
pip install -e .
```

Once the `bubble_bouncing` package is installed, use the following script to run the bubble bouncing simulation.

```
bcsim --save_folder FOLDER [-f] [--arg ARG]
```

For a full list of available arguments, see

```
bcsim -h
```
"""

from pathlib import Path
import argparse
from bubble_bouncing.utils import parse_params, available_keys
from bubble_bouncing import BounceSimulator
from bubble_bouncing import SimulationParams
from bubble_bouncing import BubbleDataVisualizer
import logging
from .config import Config

test_folder = Config().test_folder

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



if __name__=="__main__":
    main()