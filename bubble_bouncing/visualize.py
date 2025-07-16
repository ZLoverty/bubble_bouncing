import pyvista as pv
from pathlib import Path
import h5py
import yaml
import numpy as np
from scipy.interpolate import interp1d
from subprocess import run
from bubble_bouncing.bubble import SimulationParams, Bubble

class BubbleDataVisualizer(dict):
    """This class implements a few bubble simulation visualization methods. Current plan is (i) screenshot with sampled time step, (ii) 480p15 video and (iii) 720p30 video."""
    def __init__(self, folder, xlim=(-5e-3, 1e-2)):
        # s: screenshot, v: 480p video, vh: 720p video
        self.folder = Path(folder).expanduser().resolve()
        self.color = {
            0 : "#bea594",
            1 : "#d7ad9b",
            2 : "#e5dcd1",
            3 : "#e6d7d7"
        }

        # load data
        with h5py.File(self.folder / "results" / "data.h5", "r") as f:
            for key in f.keys():
                self[key] = f[key][:]

        # load params
        with open(self.folder / "params.yaml") as f:
            params = yaml.safe_load(f)
            self.params = SimulationParams(**params)

        # load mesh
        self.mesh = np.load(self.folder / "mesh.npy")
        
        self.xlim = xlim

        # crop data based on xlim
        self.ind = (self["x"][:, 0] > self.xlim[0]) * (self["x"][:, 0] < self.xlim[1])
        
        # apply xlim 
        for key in self:
            self[key] = self[key][self.ind]

        self.vis_folder = self.folder / "visualizations"

    def _interp(self, mode="v", playback=0.01):
        """Fix the play speed at 0.01X real-time. Then we can infer the required number of frames N, using the duration of the simulation T, and the output fps. This function interpolate the data based on the visualization mode.
        
        Parameters
        ----------
        mode : str
            "v" for 480p15 video, "vh" for 720p30 video
        playback : float
            the speed of video compared to real time
        
        Returns
        -------
        data : dict
            interpolated data
        """
        tmin, T = self["t"].min(), self["t"].max()
        
        if mode in ["s", "w"]:
            N = len(self["t"])
        elif mode in ["v", "vh", "custom"]:
            _, fps = self._mode_specs(mode)
            N = int((T-tmin) / playback * fps)
        else:
            raise ValueError("mode has to be 'w', 's', 'v' or 'vh'.")       
        
        data = {}
        t = np.linspace(tmin, T, N)
        data["t"] = t
        keys = [key for key in self.keys() if key != "t"]
        for key in keys:
            f = interp1d(self["t"], self[key], axis=0)
            data[key] = f(t)
        return data

    def _setup_dir(self, vis_name):
        """Folder to save screenshots."""
        if self.vis_folder.exists() == False:
            self.vis_folder.mkdir()
        folder = self.vis_folder / vis_name
        if folder.exists() == False:
            folder.mkdir()
        return folder
    
    def _mode_specs(self, mode):
        """Return frame height and frame rate for various modes."""
        if mode == "v": # low quality video
            return 480, 15
        elif mode == "vh": # high quality video
            return 720, 30
        elif mode in ["s", "w"]: # screenshot or window
            return 1080, None
        elif mode == "custom":
            return 480, 200
        else:
            raise ValueError("mode has to be 'v' or 'vh'.")
        
    def traj_com(self, mode="w", playback=0.01):
        """Center of mass trajectory."""
        vis_name = "traj_com"

        data = self._interp(mode=mode, playback=playback)

        traj = pv.PolyData(data["x"])
        height, fps = self._mode_specs(mode)
        shape = (int(height*8/9) * 2, height) # 16:9 aspect ratio

        if mode == "s":
            folder = self._setup_dir(vis_name)
            pl = pv.Plotter(off_screen=True)
            pl.add_mesh(traj, color=self.color[0])
            self.set_camera(pl)
            pl.screenshot(
                filename = folder / f"screenshot.png",
                window_size = shape
            )

        elif mode == "w": # for development
            folder = self._setup_dir(vis_name)
            pl = pv.Plotter(window_size=shape)
            sp = pv.Sphere(radius=self.params.R, center=data["x"][0])
            pl.add_mesh(
                sp, 
                color=self.color[0],
                lighting=True)
            self.set_camera(pl)
            pl.show()
        
        elif mode in ["v", "vh"]:
            folder = self._setup_dir(vis_name)
            tmp_folder = folder / "_tmp"
            tmp_folder.mkdir(exist_ok=True)
            pl = pv.Plotter(off_screen=True)
            actor = None
            for num, x in enumerate(data["x"]):
                print(f"Frame {num}", end="\r")
                sp = pv.Sphere(radius=self.params.R, center=x)
                if actor is None:
                    # the first step, just plot
                    actor = pl.add_mesh(sp, color=self.color[0])
                    self.set_camera(pl)
                else:
                    # the following steps, remove the previous actor
                    pl.remove_actor(actor)
                    actor = pl.add_mesh(sp, color=self.color[0])

                pl.screenshot(
                    filename = tmp_folder / f"{num:04d}.png",
                    window_size = shape
                )
            
            # make video
            run(["ffmpeg", "-framerate", f"{fps}", "-y",
                 "-i", tmp_folder / "%04d.png", 
                 folder / f"{height:d}p{fps:d}.mp4"])
            
            # remove screenshots
            for f in tmp_folder.iterdir():
                f.unlink()
            tmp_folder.rmdir()
        else: # show on screen 
            raise ValueError("mode has to be 'w', 's', 'v' or 'vh'.")

    def draw_surface(self, plotter):
        xlim = self.xlim
        ylim = (-1e-4, 0)
        zlim = (-0.001, 0.001)
        surface = pv.Box((*xlim, *ylim, *zlim))
        plotter.add_mesh(
            surface,
            color='steelblue',    
            show_edges=True,   
            edge_color='black',
            smooth_shading=True, 
            lighting=True         
        )

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
        theta = self.params.theta / 180 * np.pi
        up = np.array([np.sin(theta), -np.cos(theta), 0])
        plotter.camera_position = [(0, 0, -20),
                                   (0, 0, 0),
                                   up]
        self.draw_surface(plotter)
        # self.draw_reference_box(plotter, expand_factor=0.0)
        plotter.reset_camera()
        plotter.camera.zoom(1.5)
        plotter.show_axes()

    def morphology(self, mode="w", playback=0.01):
        """Bubble morphology visualization."""
        vis_name = "morphology"

        data = self._interp(mode=mode, playback=playback)

        traj = pv.PolyData(data["x"])
        height, fps = self._mode_specs(mode)
        shape = (int(height*8/9) * 2, height) # 16:9 aspect ratio

        inds = np.arange(len(data["t"]))

        if mode == "s":
            folder = self._setup_dir(vis_name)
            pl = pv.Plotter(off_screen=True)
            for ind in inds[::5]:
                points = np.column_stack([self.mesh[:, 0]+data["x"][ind, 0], data["h"][ind], self.mesh[:, 2]+data["x"][ind, 2]])
                surf = pv.PolyData(points).delaunay_2d()
                surf["height"] = data["h"][ind] - data["h"][ind].min()
                pl.add_mesh(surf, scalars="height", cmap="viridis")
            self.set_camera(pl)
            pl.screenshot(
                filename = folder / f"screenshot.png",
                window_size = shape
            )
            
        elif mode == "w": # for development          
            pl = pv.Plotter(window_size=shape)
            for ind in inds[::5]:
                points = np.column_stack([self.mesh[:, 0]+data["x"][ind, 0], data["h"][ind], self.mesh[:, 2]+data["x"][ind, 2]])
                surf = pv.PolyData(points).delaunay_2d()
                surf["height"] = data["h"][ind] - data["h"][ind].min()
                pl.add_mesh(surf, scalars="height", cmap="viridis")
            self.set_camera(pl)
            pl.show()
        
        elif mode in ["v", "vh", "custom"]:
            folder = self._setup_dir(vis_name)
            tmp_folder = folder / "_tmp"
            tmp_folder.mkdir(exist_ok=True)
            pl = pv.Plotter(off_screen=True)
            actor = None
            for ind in inds:
                print(f"Frame {ind}", end="\r")
                points = np.column_stack([self.mesh[:, 0]+data["x"][ind, 0], data["h"][ind], self.mesh[:, 2]+data["x"][ind, 2]])
                surf = pv.PolyData(points).delaunay_2d()
                surf["height"] = data["h"][ind] - data["h"][ind].min()
                if actor is None:
                    # the first step, just plot
                    actor = pl.add_mesh(surf, scalars="height", cmap="viridis")
                    self.set_camera(pl)
                else:
                    # the following steps, remove the previous actor
                    pl.remove_actor(actor)
                    actor = pl.add_mesh(surf, scalars="height", cmap="viridis")

                pl.screenshot(
                    filename = tmp_folder / f"{ind:04d}.png",
                    window_size = shape
                )
            
            # make video
            run(["ffmpeg", "-framerate", f"{fps}", "-y",
                 "-i", tmp_folder / "%04d.png", 
                 folder / f"{height:d}p{fps:d}.mp4"])
            
            # remove screenshots
            for f in tmp_folder.iterdir():
                f.unlink()
            tmp_folder.rmdir()
        else: # show on screen 
            raise ValueError("mode has to be 'w', 's', 'v' or 'vh'.")

    def Oseen_circulation(self, mode="w", playback=0.01):
        """Visualize the circulation flow around the bubble induced by the Oseen wake of the imaginary bubble."""

        def draw_bubble_surface_flow(plotter, im, re, clip=True, max_mag=0.01):
            """Use provided bubbles, draw the two bubbles and the flow around the real bubble."""
            sp_im = pv.Sphere(radius=im.a, center=im.pos)
            sp_re = pv.Sphere(radius=re.a, center=re.pos)

            actor_im = plotter.add_mesh(sp_im, opacity=0.3, smooth_shading=True, color=self.color[0])
            actor_re = plotter.add_mesh(sp_re, color=self.color[1], smooth_shading=True)

            # compute Oseen flow on real bubble
            flow = im.Oseen_wake(re.pos+re.surf_coords)
            surface_flow = (flow * re.unit_tangents).sum(axis=1, keepdims=True) * re.unit_tangents

            grid = pv.PolyData(re.pos+re.surf_coords)
            grid["flow"] = clip_vectors(surface_flow, max_mag=max_mag) if clip else surface_flow
            glyph = grid.glyph(orient="flow", scale="flow", factor=0.1)
            actor_glyph = plotter.add_mesh(glyph)
            return [actor_im, actor_re, actor_glyph]

        def clip_vectors(vectors, max_mag=0.01):
            """Set the maximum vector size to be 0.01 to avoid extremely large arrows that block the view.
            
            Parameters
            ----------
            vectors : ndarray
                N x 3 array of vectors to clip
            max_mag : float
                maximum allow magnitude, vectors beyond this will be limited to this value
            
            """
            norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            inds = (norm > max_mag).squeeze()
            clipped = vectors.copy()
            clipped[inds] *=  max_mag / norm[inds]
            return clipped

        vis_name = "Oseen_circulation"

        data = self._interp(mode=mode, playback=playback)
        # import pdb
        # pdb.set_trace()
        height, fps = self._mode_specs(mode)
        shape = (int(height*8/9) * 2, height) # 16:9 aspect ratio

        inds = np.arange(len(data["t"]))

        # determine the time for the snapshot
        first_valid_index = np.argmax(~np.isnan(data["x_im"][:, 0]))
        t_first_bounce = data["t"][first_valid_index]
        # look at 5 ms after the first bounce
        t_snap = t_first_bounce + 2e-3
            
        ind_snap = np.abs(data["t"]-t_snap).argmin()

        if mode == "w": # for development          
            pl = pv.Plotter(window_size=shape)
            im = Bubble(self.params.R, U=data["V_im"][ind_snap])
            im.set_pos(data["x_im"][ind_snap])
            re = Bubble(self.params.R, U=data["V"][ind_snap])
            re.set_pos(data["x"][ind_snap])
            draw_bubble_surface_flow(pl, im, re)
            self.set_camera(pl)
            pl.show()

        elif mode == "s":
            folder = self._setup_dir(vis_name)
            pl = pv.Plotter(window_size=shape, off_screen=True)
            im = Bubble(self.params.R, U=data["V_im"][ind_snap])
            im.set_pos(data["x_im"][ind_snap])
            re = Bubble(self.params.R, U=data["V"][ind_snap])
            re.set_pos(data["x"][ind_snap])
            draw_bubble_surface_flow(pl, im, re)
            self.set_camera(pl)
            pl.screenshot(
                filename = folder / f"screenshot.png"
            )

        elif mode in ["v", "vh", "custom"]:
            folder = self._setup_dir(vis_name)
            tmp_folder = folder / "_tmp"
            tmp_folder.mkdir(exist_ok=True)
            pl = pv.Plotter(off_screen=True)
            actor = None

            for ind in inds:
                print(f"Frame {ind}", end="\r")
                pl = pv.Plotter(window_size=shape, off_screen=True)
                im = Bubble(self.params.R, U=data["V_im"][ind])
                im.set_pos(data["x_im"][ind])
                re = Bubble(self.params.R, U=data["V"][ind])
                re.set_pos(data["x"][ind])

                actors = None
                if actors is None:
                    # the first step, just plot
                    actors = draw_bubble_surface_flow(pl, im, re)
                    self.set_camera(pl)
                else:
                    # the following steps, remove the previous actor
                    for actor in actors:
                        pl.remove_actor(actor)
                    actors = draw_bubble_surface_flow(pl, im, re, max_mag=0.05)

                pl.screenshot(
                    filename = tmp_folder / f"{ind:04d}.png"
                )
            
            # make video
            run(["ffmpeg", "-framerate", f"{fps}", "-y",
                 "-i", tmp_folder / "%04d.png", 
                 folder / f"{height:d}p{fps:d}.mp4"])
            
            # remove screenshots
            for f in tmp_folder.iterdir():
                f.unlink()
            tmp_folder.rmdir()
        else: # show on screen 
            raise ValueError("mode has to be 'w', 's', 'v' or 'vh'.")
        
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(f"Bubble visualizer.")
    parser.add_argument("--folder", type=str, default="~/Documents/test", help="Main data folder.")
    parser.add_argument("--mode", type=str, default="none", help="Visualization mode: s, v or vh.")
    args = parser.parse_args()

    vis = BubbleDataVisualizer(args.folder)
    # vis.traj_com(mode=args.mode)
    # vis.morphology(mode=args.mode)
    vis.Oseen_circulation(mode=args.mode)