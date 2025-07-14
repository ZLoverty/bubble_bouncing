import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from .units import Units

class Simulator:
    """This class implements a framework for simple numerical simulations. It includes routine operations such as read and save parameters, setup directories, and logging. A specific simulation logic can be implemented by a subclass, which overrides the `pre_run`, `_run` and `post_run` methods."""
    def __init__(self, 
                 save_folder : str, 
                 params: dataclass,
                 exist_ok : bool = False):
        """Initialize the simulator with a save folder and a flag to indicate whether to overwrite existing directories.

        Parameters
        ----------
        save_folder : str or Path
            The folder where the simulation results and parameters will be saved. It can be a relative or absolute path.
        params : SimulationParams
            A dataclass object containing the simulation parameters.
        exist_ok : bool, optional
            If True, the existing directories will not raise an error. If False, an error will be raised if the directories already exist. Default is False.
        """
        self.params = params
        self.save_folder = Path(save_folder).expanduser().resolve()
        self.setup_dirs(exist_ok)
        
    def __repr__(self):
        try:
            return yaml.dump(asdict(self.params))
        except AttributeError as e:
            return f"WARNING: {e}. Use `set_params` to create parameter property"
        except TypeError:
            return yaml.dump(self.params)
    
    def setup_dirs(self, exist_ok):
        
        self.params_file = self.save_folder / "params.yaml"
        self.log_file = self.save_folder / "sim.log"
        self.data_dir = self.save_folder / "results"
        self.save_folder.mkdir(parents=True, exist_ok=exist_ok)
        self.data_dir.mkdir(exist_ok=exist_ok)
        
    def pre_run(self):
        raise NotImplementedError("pre_run() has not been implemented.")
        
    def post_run(self):
        raise NotImplementedError("post_run() has not been implemented.")
        
    def set_params(self, params):
        """Set parameter properties.
        params -- a dataclass object specifying the set of parameters used in the simulation."""
        self.params = params
    
    def load_params(self):
        """Read args from file `params.yaml`."""
        try:
            with open(self.params_file, "r") as f:
                params_dict = yaml.safe_load(f)
                self.params = type(self.params)(**params_dict)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e}")

    def save_params(self):
        try:
            with open(self.params_file, "w") as f:
                yaml.dump(asdict(self.params), f)
        except AttributeError as e:
            raise AttributeError(f"Attribute error: {e}")
    
    def _run(self):
        raise NotImplementedError("_run() has not been implemented.")

    def run(self):
        self.pre_run()
        self._run()
        self.post_run()