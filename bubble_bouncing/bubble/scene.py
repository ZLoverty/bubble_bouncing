from pyvistaqt import BackgroundPlotter
import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d
from PyQt5.QtCore import QTimer

class Scene(BackgroundPlotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear_all()

    def add(self, mesh, data):
        """Add mesh as actor to scene. 

        Parameters
        ----------
        mesh : pyvista.DataSet
            The mesh to be added.
        data : dict
            Trajectory data, must contain 't' and 'x'.
        
        Returns
        -------
        actor : pv.Actor
            The actor object.
        """
        t, x = self._read_data(data)
        # update time range
        self.T = max(self.T, t.max())
        self.t0 = min(self.t0, t.min())
        # add actor
        actor = self.add_mesh(mesh)
        actor_data = (actor, data)
        self.actor_list.append(actor_data)
        # self.interp_actor_list = self.actor_list.copy()
        return actor

    def _read_data(self, data):
        """Validate the data: (i) contains 'x' and 't', (ii) 'x' has shape (N, 3), (iii) t.shape[0] = x.shape[0]."""
        if not all(k in data for k in ["t", "x"]):
            raise ValueError("Input data dictionary must contain 't' and 'x'.")
        
        t = np.asarray(data["t"])
        x = np.asarray(data["x"])

        # ensure position data shape
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("The position data shape must be (N, 3).")
        
        # ensure t and x are the same length
        assert x.shape[0] == t.shape[0]

        return t, x

    def _interp(self):
        """Generate interpolation constructor based on input data.
        
        Parameters
        ----------
        data : dict
            Trajectory data, must contain 't' and 'x'.
        
        Returns
        -------
        interp_x : scipy.interpolate.interp1d
        """
        # validate input data structure
        self.interp_actor_list = []
        for actor, data in self.actor_list:
            t, x = self._read_data(data)
            f = interp1d(t, x, axis=0, fill_value="extrapolate")
            x_interp = f(self.t)
            self.interp_actor_list.append((actor, {"t": self.t, "x": x_interp}))

    def set_time(self, t):
        """Set the time points to prepare the video. Update the 'data_interp' field with interpolated data.
        
        Parameters
        ----------
        t : array_like[float]
            Time sequence to evaluate the object positions.
        """
        self.t = t
        self._interp()
        
    
    def set_time_uniform(self, fps=30, playback=0.01):
        """Set time sequence uniformly according to fps and playback speed.
        
        Parameters
        ----------
        fps : float
            Frame per second.
        playback : float
            Playback speed compared to real time.
        """

        nFrame = int((self.T - self.t0) / playback * fps )
        self.t = np.linspace(self.t0, self.T, nFrame)
        self._interp()

    def play(self, t_range=None, fps=30):
        """Play animation.
        
        Parameters
        ----------
        t_range : Sequence
            [tmin, tmax], the animation is limited by this range.  
        """
        t = self.t
        delay = 1 / fps
        if t_range is None:
            ind = np.ones_like(t).astype(bool)
        else:
            if len(t_range) == 2:
                ind = np.logical_and(t>t_range[0], t<=t_range[1])
            else:
                raise ValueError("t_range must be a tuple or list of two floats.")
        step = [0]
        timer = QTimer()
        def update():
            if step[0] >= len(t)-1:
                timer.stop()
                print("Animation complete")
                return
            for actor, data in self.interp_actor_list:
                new_position = data["x"][ind][step[0]]
                actor.SetPosition(new_position)
            self.render()
            timer.start(delay)
            step[0] += 1
        
        timer.timeout.connect(update)
        # delay = int(delays[step[0]]*1000)
        timer.start(0)

    def clear_all(self):
        """Clear actor list and time."""
        try:
            l = getattr(self, "actor_list")
            for actor, _ in l:
                self.remove_actor(actor)
        except AttributeError:
            pass
        self.actor_list = []
        self.T = 0
        self.t0 = 0
        self.t = None # The time sequence