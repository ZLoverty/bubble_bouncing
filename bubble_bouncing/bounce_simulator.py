"""
main.py
=======

The main logic of the bubble bouncing simulation. It initializes the simulator, sets up the parameters, and runs the simulation.

"""
import numpy as np
from scipy import integrate
import h5py
from bubble_bouncing.simulation import Simulator, Units, DataBuffer
from bubble_bouncing.bubble import SimulationParams, compute_tff, compute_drag, compute_amf, compute_buoyancy, compute_lift, Bubble
from bubble_bouncing.utils import _decomp, gradient_operators
import logging

class BounceSimulator(Simulator):

    def pre_run(self):
        logging.info("============NEW SIMULATION==========")
        logging.info(f"R = {self.params.R*1e3:.2f} mm | angle = {self.params.theta:.1f} deg | Cl = {self.params.lift_coef:.1f}")
        self.save_params()
        self.units = self._setup_units()
        self._setup_mesh()
        self.Grad = gradient_operators(self.params.N, self.dx)
        self._setup_index_masks()
        self.initial_state = self._initial_condition()
        
        self.print_interval = self.units.to_nondim(self.params.print_time, "time")
        self.save_interval = self.units.to_nondim(self.params.save_time, "time")
        self.last_print = 0.0
        self.last_save = 0.0
        self.first_bounce = False
        self.data_to_store = [
            ("t", ()),
            ("h", (self.params.N**2, )),
            ("V", (3, )),
            ("x", (3, )), 
            ("buoyancy", (3, )),
            ("drag", (3, )),
            ("amf2", (3, )),
            ("tff", (3, )),
            ("lift", (3, )),
            ("x_im", (3, )),
            ("V_im", (3, ))
        ]
        self._setup_databuffers()
        self.im = None
        
    def _setup_units(self):
        """Setup scales to nondimensionalize the equations."""
        logging.info("Setting units")
        params = self.params
        R = params.R
        mu = params.mu
        sigma = params.sigma
        V = sigma / mu
        scales = {
            "length": (R, "m"),
            "velocity": (V, "m/s"),
            "time": (R/V, "s"),
            "pressure": (sigma/R, "N/m^2"),
            "acceleration": (V**2/R, "m/s^2")
        }
        return Units(scales)
    
    def _setup_mesh(self):
        """Setup a NxN square mesh."""
        logging.info("Setting mesh")
        rm = self.params.rm
        N = self.params.N
        x = np.linspace(-rm, rm, num=N)
        z = np.linspace(-rm, rm, num=N)
        X, Z = np.meshgrid(x, z) 
        X = X.flatten()
        Z = Z.flatten()
        self.mesh = np.column_stack([X, np.zeros_like(X), Z])
        self.dx = x[1] - x[0] # for computing gradient operators
        # save mesh
        mesh_dim = self.units.to_dim(self.mesh, "length")
        np.save(self.save_folder / "mesh.npy", mesh_dim)
        
    def _setup_index_masks(self):
        """Meanwhile get a few index masks (center, edge, center slice) for monitoring the simulation progress."""
        N = self.params.N
        mask = np.zeros((N, N)).astype(bool)
        center_ind, edge_ind, centerslice_ind = mask.copy(), mask.copy(), mask.copy()
        center_ind[N//2, N//2] = True
        self.center_ind = center_ind.flatten()
        edge_ind[:, 0], edge_ind[:, -1], edge_ind[0, :], edge_ind[-1, :] = True, True, True, True
        self.edge_ind = edge_ind.flatten()
        centerslice_ind[N//2, :] = True
        self.centerslice_ind = centerslice_ind.flatten()

    def _initial_condition(self):
        """Initialize the simulation with the bubble located H0 away from the surface in y-direction. """
        logging.info("Setting initial conditions")
        H0_dim = self.params.H0
        H0 = self.units.to_nondim(H0_dim, "length")
        h0 = H0 + (self.mesh[:, 0]**2 + self.mesh[:, 2]**2) / 2
        V0_dim = self.params.V0
        V0 = self.units.to_nondim(V0_dim, "velocity")
        x0 = np.array([0, H0+1, 0])
        theta_rad = self.params.theta / 180 * np.pi
        V0v = np.array([-V0*np.sin(theta_rad), V0*np.cos(theta_rad), 0])
        return np.concatenate([h0, V0v, x0])

    def _setup_databuffers(self):
        # save t, h, V, x
        h5file = self.data_dir / "data.h5"
        with h5py.File(h5file, "w") as f:
            pass
        self.data_buffer = {}
        for name, shape in self.data_to_store:
            self.data_buffer[name] = DataBuffer(h5file, name, shape)
    
    def compute_forces(self, t, y):
        """Compute the forces acting on a bubble."""

        # constants
        R = self.params.R
        rho = self.params.rho
        mu = self.params.mu
        gs = self.params.g
        theta_rad = self.params.theta / 180 * np.pi
        g = np.array([-gs*np.sin(theta_rad), gs*np.cos(theta_rad), 0])
        dx = self.dx
        units = self.units

        # load current state
        h, V, x = _decomp(y)
        p = self.YL_equation(h)
        h_dim = units.to_dim(h, "length")
        dx_dim = units.to_dim(dx, "length")
        V_dim = units.to_dim(V, "velocity")
        p_dim = units.to_dim(p, "pressure")
        dhdx = self.Grad["x"] @ h
        H = h_dim[self.center_ind][0]  # height at the center of the bubble
        buoyancy = compute_buoyancy(R, rho, g)
        drag = compute_drag(R, V_dim, rho, mu)
        amf2, Cm = compute_amf(H, R, rho, V_dim)
        tff = compute_tff(p_dim, dhdx, dx_dim)

        
        if self.first_bounce:
            # execute this only after the first bounce
            t_dim = units.to_dim(t, "time")
            x_im = self.x_im_start + (t_dim - self.t_im) * self.U_im
            x_re = units.to_dim(x, "length")
            U_re = units.to_dim(V, "velocity")
            self.im.set_pos(x_im)
            self.re.set_pos(x_re)
            flow = self.im.Oseen_wake(self.re.pos+self.re.surf_coords)
            surface_flow = flow * self.re.unit_tangents
            lift = compute_lift(self.im.a, surface_flow, self.re.ds, U_re, lift_coef=self.params.lift_coef)
        else:
            lift = np.zeros(3) * np.nan

        return {
            "buoyancy": buoyancy,
            "drag": drag,
            "amf2": amf2,
            "tff": tff,
            "Cm": Cm,
            "lift": lift
        }

    def post_run(self):
        logging.info("Simulation finished.")

    def YL_equation(self, h):
        """Implement the dimensionless Young-Laplace equation
        
        p = 2 - d2h

        with boundary condition p = 0. Note that we initialize the height profile as h0 = H0 + (x^2 + z^2) / 2, which automatically gives all 0 profile at the beginning.
        """
        p = 2 - self.Grad["2x"] @ h - self.Grad["2z"] @ h
        p[self.edge_ind] = 0
        return p
    
    def _run(self):
        def film_drainage(t, y):
            """Implement the dimensionless film drainage equation.
            
            dhdt = U * dhdx + 1/3 * dx(d2pdx * h^3) + 1/3 * dz(d2pdz * h^3) 
            
            where and d, d2 denotes the gradient operator to the first and second order. 

            The boundary condition of dhdt is determined by a force balance equation, which gives the rate of change of bubble velocity. Basically,

            m * dVdt = sum(forces)

            The resulting V[1] is the boundary condition of dhdt at each time step.
            """
            h, V, x = _decomp(y)
            Gx, Gz = self.Grad["x"], self.Grad["z"]

            p = self.YL_equation(h)

            dhdt = (
                V[0] * Gx @ h
                + 1/3 * Gx @ ((Gx @ p) * h**3)
                + 1/3 * Gz @ ((Gz @ p) * h**3)
            )

            dhdt[self.edge_ind] = V[1]

            forces = self.compute_forces(t, y)

            R_dim = self.params.R
            rho = self.params.rho
            dVdt_dim = (forces["buoyancy"] + forces["drag"] + forces["amf2"] + forces["tff"]) / (4/3 * np.pi * R_dim**3 * rho * forces["Cm"])
            dVdt = self.units.to_nondim(dVdt_dim, "acceleration")

            return np.concatenate([dhdt, dVdt, V])
        
        def event_print(t, y):
            """This is a workaround of the action we take periodically during the simulation. We need this because the time stepping in this code is implicit, meaning that we do not have a "loop", but rather one line of code. This function is passed as an argument to the `scipy.integrate.solve_ivp()` function. It checks the time and state, and can perform info print, and data flushing."""

            # Print info
            h, V, x = _decomp(y)

            if t == 0 or t - self.last_print >= self.print_interval:
                self.last_print = t  
                t_dim = self.units.to_dim(t, "time")
                h_dim = self.units.to_dim(h, "length")
                V_dim = self.units.to_dim(V, "velocity")
                x_dim = self.units.to_dim(x, "length")
                # print info
                logging.debug(
                    f"t={t_dim*1e3:.2f} ms | y={x_dim[1]*1e3:.2f} mm | Vy={V_dim[1]*1e3:.1f} mm/s"
                )

                # buffer data
                self.data_buffer["t"].append(t_dim)
                self.data_buffer["h"].append(h_dim)
                self.data_buffer["V"].append(V_dim)
                self.data_buffer["x"].append(x_dim)

                forces = self.compute_forces(t, y)
                del forces["Cm"]
                for name, force in forces.items():
                    self.data_buffer[name].append(force)
                if self.im is None:
                    self.data_buffer["x_im"].append(np.zeros(3) * np.nan)
                    self.data_buffer["V_im"].append(np.zeros(3) * np.nan)
                else:
                    self.data_buffer["x_im"].append(self.im.pos)
                    self.data_buffer["V_im"].append(self.im.U)

            # Flush data to file
            if t == 0 or t - self.last_save >= self.save_interval:
                t_dim = self.units.to_dim(t, "time")
                logging.info(f"Dumping data to file at t = {t_dim*1e3:.2f} ms")
                self.last_save = t
                for name, _ in self.data_to_store:
                    self.data_buffer[name].flush()

            # detect first bounce
            if x[1] < 1 and self.first_bounce == False:
                units = self.units
                R = self.params.R
                self.first_bounce = True
                self.U_im = units.to_dim(V, "velocity")
                self.t_im = units.to_dim(t, "time")
                logging.info(f"First bounce at {self.t_im*1e3:.1f} ms")
                self.x_im_start = units.to_dim(x, "length")
                logging.debug(f"First bounce location y={x[1]}")
                self.im = Bubble(R, U=self.U_im)
                self.im.set_pos(self.x_im_start)
                self.re = Bubble(R)
                lift = np.array([0, 0, 0])

            return 1

        logging.info("Simulation starts!")
        T = self.units.to_nondim(self.params.T, "time")

        sol = integrate.solve_ivp(film_drainage, [0, T], self.initial_state, method="BDF", events=event_print, atol=1e-6, rtol=1e-3)

    def run(self):
        self.pre_run()
        self._run()
        self.post_run()

if __name__ == "__main__":

    params = SimulationParams(R=6e-4, N=50)
    sim = BounceSimulator("~/Documents/.test", params, exist_ok=True)
    
    sim.run()