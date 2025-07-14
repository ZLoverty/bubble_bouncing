import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import yaml
import numpy as np
from scipy.interpolate import interp1d

class BubbleDataVisualizer:
    def __init__(self, folder, interp=True, dt=1e-4, xlim=(-5e-3, 1e-2)):
        folder = Path(folder).expanduser().resolve()
        with h5py.File(folder / "results" / "data.h5", "r") as f:
            self.t = f["t"][:]
            self.x = f["x"][:]
            self.h = f["h"][:]
        with open(folder / "params.yaml") as f:
            self.params = yaml.safe_load(f)
        self.mesh = np.load(folder / "mesh.npy")

        if interp:
            self._interp(dt=dt)
        
        self.xlim = xlim
        # crop data based on xlim
        self.ind = (self.x[:, 0] > self.xlim[0]) * (self.x[:, 0] < self.xlim[1])

    def _interp(self, dt=1e-4):
        t = self.t
        N = int((t.max()-t.min()) / dt)
        # new t
        self.t = np.linspace(t.min(), t.max(), N)
        f = interp1d(t, self.x, axis=0)
        self.x = f(self.t)
        f = interp1d(t, self.h, axis=0)
        self.h = f(self.t)

    def traj_com(self):
        
        traj = pv.PolyData(self.x[self.ind])
        
        cmap = plt.get_cmap("tab10")
        pl = pv.Plotter()

        pl.add_mesh(traj, color=cmap(0))
        self.set_camera(pl)
        
        pl.show()

    def draw_surface(self, plotter):
        xlim = self.xlim
        ylim = (-1e-4, 0)
        zlim = (-0.001, 0.001)
        surface = pv.Box((*xlim, *ylim, *zlim))
        plotter.add_mesh(
            surface,
            color='steelblue',       # Set a solid color
            show_edges=True,         # Show the edges to define the box shape
            edge_color='black',      # Make edges black for better contrast
            smooth_shading=True,     # Smooth the appearance of the faces
            lighting=True            # Ensure lighting is enabled for 3D appearance
        )
        # plotter.add_ruler(
        #     pointa=(xlim[0], ylim[0], 0.0),
        #     pointb=(xlim[1], ylim[0], 0.0)
        # )
    def draw_reference_box(self, plotter, expand_factor=.1):
        xmin, xmax, ymin, ymax, zmin, zmax = tuple(plotter.bounds)
        xl = xmax - xmin
        yl = ymax - ymin
        zl = zmax - zmin
        xmin -= expand_factor * xl
        ymin -= expand_factor * yl
        zmin -= expand_factor * zl
        xmax += expand_factor * xl
        ymax += expand_factor * yl
        zmax += expand_factor * zl
        box = pv.Box((xmin, xmax, ymin, ymax, zmin, zmax))
        plotter.add_mesh(box, style='wireframe', color='black', line_width=2)

    def set_camera(self, plotter):
        theta = self.params["theta"] / 180 * np.pi
        up = np.array([np.sin(theta), -np.cos(theta), 0])
        plotter.camera_position = [(0, 0, -20),
                                   (0, 0, 0),
                                   up]
        self.draw_surface(plotter)
        self.draw_reference_box(plotter, expand_factor=0.0)
        plotter.reset_camera()
        # pl.camera.zoom(.7)
        plotter.show_axes()

    def film_morphology(self, nSample=5):
        inds = np.arange(len(self.t[self.ind]))
        selected = np.random.choice(inds, size=nSample, replace=False)
        pl = pv.Plotter()
        for ind in selected:
            points = np.column_stack([self.mesh[:, 0]+self.x[ind, 0], self.h[ind], self.mesh[:, 2]+self.x[ind, 2]])
            surf = pv.PolyData(points).delaunay_2d()
            surf["height"] = self.h[ind]
            pl.add_mesh(surf, scalars="height", cmap="viridis", show_edges=True)

        self.set_camera(pl)
        pl.show()

    def Oseen_circulation(self, nSample=5):
        """Visualize the circulation flow around the bubble induced by the Oseen wake of the imaginary bubble."""
        pass
if __name__=="__main__":
    vis = BubbleDataVisualizer("~/Documents/BC_simulation/test")
    vis.traj_com()
    # vis.film_morphology()