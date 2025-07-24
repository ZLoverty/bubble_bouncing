from bubble_bouncing import Scene
import numpy as np
import pyvista as pv
from pathlib import Path
import h5py
import yaml


self = Scene(toolbar=False, menu_bar=False)

class Helloworld(Scene):
    def construct(self):
        folder = Path("/Users/zhengyang/Documents/.bcsim_test")
        data_file = folder / "results" / "data.h5"
        data = {}
        with h5py.File(data_file, "r") as f:
            for key in f.keys():
                data[key] = f[key][:]
        params_file = folder / "params.yaml"
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)
        theta = params["theta"] / 180 * np.pi
        sp = self.add(pv.Sphere(radius=params["R"], center=data["x"][0]), data)
        self.set_time_uniform()
        self.camera_position = [
            (0.00015806665890375784, 0.007864070875924164, -0.010385471243909032),
            (0.00015806665890375784, 0.007864070875924164, 0.0),
            (np.sin(theta), -np.cos(theta), 0.0)]
        self.show_axes()
        self.play()

if __name__=="__main__":
    h = Helloworld()
    h.construct()
