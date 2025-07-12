"""
main.py
=======

The main logic of the bubble bouncing simulation. It initializes the simulator, sets up the parameters, and runs the simulation.

"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from simulation import Simulator, Units
from bubble import SimulationParams, compute_tff, compute_drag, compute_amf, compute_buoyancy
from utils import _decomp

class BounceSimulator(Simulator):

    def pre_run(self):
        self._setup_mesh()
        self._setup_index_masks()
        self._initial_condition()
        self._setup_canvas()
        # self.update_vis_data(0, self.initial_state)
        # self.update_canvas()
        self._setup_gradient_operators()
        self.print_interval = self.units.to_nondim(self.params.save_time, "time")
        self.last_print = 0.0

    def _setup_mesh(self):
        """Setup a NxN square mesh."""
        rm = self.params.rm
        N = self.params.N
        x = np.linspace(-rm, rm, num=N)
        z = np.linspace(-rm, rm, num=N)
        X, Z = np.meshgrid(x, z) 
        X = X.flatten()
        Z = Z.flatten()
        self.mesh = np.column_stack([X, np.zeros_like(X), Z])
        self.dx = x[1] - x[0] # for computing gradient operators

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
        H0_dim = self.params.H0
        H0 = self.units.to_nondim(H0_dim, "length")
        h0 = H0 + (self.mesh[:, 0]**2 + self.mesh[:, 2]**2) / 2
        V0_dim = self.params.V0
        V0 = self.units.to_nondim(V0_dim, "velocity")
        theta_rad = self.params.theta / 180 * np.pi
        V0v = np.array([-V0*np.sin(theta_rad), V0*np.cos(theta_rad), 0])
        self.initial_state = np.concatenate([h0, V0v])

    def _setup_canvas(self, y0):
        if plt.gca() is not None:
            self.fig = plt.gcf()
            self.ax = plt.gca()
            self.ax.cla()
        else:
            self.fig, self.ax = plt.subplots()
        self.ax.set_title("Bubble simulation")
        self.ax.set_xlabel("$t$")
        self.ax.set_ylabel("$y$")
        rm_dim = self.units.to_dim(self.params.rm, "length")
        H0_dim = self.params.H0
        self.ax.set_xlim(-rm_dim, rm_dim)
        self.ax.set_ylim(0, H0_dim)

        h, V, x = _decomp(y0)

        self.line, = self.ax.plot([], [], "o")
        self.annotation_text = self.ax.annotate(f"0", (0.9, 0.9), xycoords="axes fraction")
        plt.ion()
        plt.show()
        plt.pause(.1)

    def _setup_gradient_operators(self):
        """Setup gradient operators as sparse matrices."""
        from scipy.sparse import diags, kron, identity
        N = self.params.N
        # 1D first derivative operator (2nd order accuracy)
        Dx = diags([-1, 1], [-1, 1], shape=(N, N))
        Dx = Dx.toarray()
        Dx[0, :3] = [-3, 4, -1]
        Dx[-1, -3:] = [1, -4, 3]
        Dx = Dx / (2*self.dx)
        # 1D second derivative operator (2nd order accuracy)
        D2x = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
        D2x = D2x.toarray()
        D2x[0, :4] = [2, -5, 4, -1]
        D2x[-1, -4:] = [-1, 4, -5, 2]
        D2x = D2x / (self.dx)**2
        # 1st and 2nd order derivative operators
        eye = identity(N)
        Gx = kron(eye, Dx)
        Gz = kron(Dx, eye)
        G2x = kron(eye, D2x)
        G2z = kron(D2x, eye)
        # 2D Laplacian operator
        L2D = G2x + G2z
        self.Grad = {"x": Gx, "z": Gz, "2x": G2x, "2z": G2z}

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

        # load current state
        h, V, x = _decomp(y)
        p = self.YL_equation(h)
        h_dim = self.units.to_dim(h, "length")
        dx_dim = self.units.to_dim(dx, "length")
        V_dim = self.units.to_dim(V, "velocity")
        p_dim = self.units.to_dim(p, "pressure")
        dhdx = self.Grad["x"] @ h
        H = h_dim[self.center_ind][0]  # height at the center of the bubble
        buoyancy = compute_buoyancy(R, rho, g)
        drag = compute_drag(R, V_dim, rho, mu)
        amf2, Cm = compute_amf(H, R, rho, V_dim)
        tff = compute_tff(p_dim, dhdx, dx_dim)

        return {
            "buoyancy": buoyancy,
            "drag": drag,
            "amf2": amf2,
            "tff": tff,
            "Cm": Cm,
        }

    def update_canvas(self):
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)
        # self.ax.relim()
        # self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_vis_data(self, t, y):
        rm = self.params.rm
        N = self.params.N
        h, V, x = _decomp(y)
        t_dim = self.units.to_dim(t, "time")
        rm_dim = self.units.to_dim(rm, "length")
        h_dim = self.units.to_dim(h, "length")

        self.annotation_text.set_text(f"{t_dim*1e3:.1f} ms")
        self.xdata = np.linspace(-rm_dim, rm_dim, N)
        self.ydata = h_dim[self.centerslice_ind]

    def post_run(self):
        print("Simulation finished.")

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
            if t - self.last_print >= self.print_interval:
                h, V, x = _decomp(y)
                t_dim = self.units.to_dim(t, "time")
                h_dim = self.units.to_dim(h, "length")
                V_dim = self.units.to_dim(V, "velocity")
                print(f"t={t_dim*1e3:.2f} ms | hmin={h_dim.min()*1e3:.2f} mm | V_y={V_dim[1]*1e3:.1f} mm/s")
                self.last_print = t
                self.update_vis_data(t, y)
                self.update_canvas()
            return 1

        
        T = self.units.to_nondim(self.params.T, "time")

        sol = integrate.solve_ivp(film_drainage, [0, T], self.initial_state, method="BDF", events=event_print, atol=1e-6, rtol=1e-3)


    def run(self):
        self.pre_run()
        self._run()
        self.post_run()

if __name__ == "__main__":

    params = SimulationParams(R=6e-4, N=50)
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
    units = Units(scales)

    sim = BounceSimulator("~/Documents/test", params, units, exist_ok=True)
    sim.run()
