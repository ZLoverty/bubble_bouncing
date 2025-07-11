"""
bounce.py
=========

Description
-----------

Rewrite the bubble simulation in object oriented way. We aim to improve the modularity and readability of the simulator.

"""
import numpy as np
from scipy import integrate
import healpy as hp
from base import Simulator
from params import SimulationParams
import matplotlib.pyplot as plt
from units import Units

class Bubble:
    """Compute the forces and the flow field associated with a moving bubble in a liquid. The forces we consider are buoyancy, drag, added mass force and thin film force when the bubble touches a solid surface. The flow field we consider is an Oseen wake. The flow field is characterized by two regions: a Stokeslet in the low Reynolds region and a compensating flow """
    def __init__(self, a, U=0, rho=1e3, mu=1e-3):
        self.a = a # bubble radius
        self.U = U # upward velocity (only upward!)
        self.rho = rho # density
        self.mu = mu # viscosity
        self.pos = np.array([0, 0, 0]) # bubble position
        self.surf_coords, self.unit_normals, self.ds = self._compute_surface_coords()
        self.unit_tangents = self._compute_surface_tangent_xz()

    def Oseen_wake(self, points):
        """Compute Oseen wake at given (relative) points. 

        Args:
        points -- should be an array of (npts, 3), the points to evaluate Oseen wake flow field.

        Returns: 
        flow -- the flow velocity at each given point, also (npts, 3).
        """
        U = self.U
        a = self.a
        rho = self.rho
        mu = self.mu
        Re = rho * a * U / mu

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = (x**2 + y**2 + z**2) ** 0.5
        sint = (x**2 + y**2)**0.5 / r
        cost = z / r
        sinp = y / (x**2 + y**2)**0.5
        cosp = x / (x**2 + y**2)**0.5

        u_r = U * (
            - a**3 * cost / 2 / r**3 
            + 3 * a**2 / 2 * r**2 * Re * (
                1 - np.exp(- r * Re / 2 / a * (1 + cost))
            )
            - 3 * a * (1 - cost) / 4 / r * np.exp(- r * Re / 2 / a * (1 + cost))
        )
        u_t = U * (
            - a**3 * sint / 4 / r**3 
            - 3 * a * sint / 4 / r * np.exp(- r * Re / 2 / a * (1 + cost))
        )
        u_p = 0

        u_x = u_r * sint * cosp + u_t * cost * cosp - u_p * sinp
        u_y = u_r * sint * sinp + u_t * cost * sinp + u_p * cosp
        u_z = u_r * cost - u_t * sint

        # The velocity diverges at r=0, implying that this velocity should be considered as "far field" velocity
        # thus, we mask out all the velocities inside the imaginary shpere, where r <= a
        invalid = r <= a
        u_x[invalid] = 0
        u_y[invalid] = 0
        u_z[invalid] = 0

        return np.stack([u_x, u_y, u_z], axis=-1)
    
    def get_pos(self):
        return self.pos
    
    def set_pos(self, pos):
        self.pos = np.array(pos)

    def set_velocity(self, U):
        self.U = U
        
    def _compute_surface_coords(self, nside=4):
        """Compute the coordinates of the surface differential area and surface unit normal vectors.
        
        Args:
        nside -- controls how many parts the spherical surface is to be divided, utilizing the `healpy` package. Has to be power of 2. The number of parts will be 12*nside**2.
        
        Returns:
        surface_coords -- coordinates of surface differential area __relative to the center__ of the sphere.
        unit_normals -- unit normal vectors corresponding to the surface locations.
        differential_surface_area -- the area of each differential surface unit."""

        R = self.a
        npix = hp.nside2npix(nside)

        # Get spherical coordinates (theta, phi) of each pixel center
        theta, phi = hp.pix2ang(nside, np.arange(npix))

        x = R * np.sin(theta) * np.cos(phi) 
        y = R * np.sin(theta) * np.sin(phi) 
        z = R * np.cos(theta)
        surface_coords = np.stack([x, y, z], axis=-1)
        unit_normals = surface_coords / np.linalg.norm(surface_coords, axis=-1, keepdims=True)
        differential_surface_area = 4 * np.pi * R**2 / npix
        return surface_coords, unit_normals, differential_surface_area
    
    def _compute_surface_tangent_xz(self):
        """Compute sphere surface tangential unit vectors, specifically their projections on the xz plane."""
        tangent = np.zeros_like(self.unit_normals)
        tangent[:, 0] = self.unit_normals[:, 2]
        tangent[:, 2] = - self.unit_normals[:, 0]
        return tangent / np.linalg.norm(tangent, axis=1, keepdims=True)

class BounceSimulator(Simulator):
    def __init__(self, folder, exist_ok=False):
        super().__init__(folder, exist_ok)

    def pre_run(self, **new_params):
        self.set_params(SimulationParams())
        self.update_params(**new_params)
        self._setup_units()
        self._setup_mesh()
        self._setup_index_masks()
        self._initial_condition()
        self._setup_canvas()
        self.update_vis_data(0, self.initial_state)
        self.update_canvas()
        self._setup_gradient_operators()
        self.print_interval = self.units.to_nondim(self.params.control.save_time, "time")
        self.last_print = 0.0
        
    def _setup_canvas(self):
        if plt.gca() is not None:
            self.fig = plt.gcf()
            self.ax = plt.gca()
            self.ax.cla()
        else:
            self.fig, self.ax = plt.subplots()
        self.ax.set_title("Bubble simulation")
        self.ax.set_xlabel("$t$")
        self.ax.set_ylabel("$y$")
        rm_dim = self.units.to_dim(self.params.control.rm, "length")
        H0_dim = self.params.initial.H0
        self.ax.set_xlim(-rm_dim, rm_dim)
        self.ax.set_ylim(0, H0_dim)
        self.line, = self.ax.plot([], [], "o")
        self.annotation_text = self.ax.annotate(f"0", (0.9, 0.9), xycoords="axes fraction")
        plt.ion()
        plt.show()
        plt.pause(.1)

    def _setup_units(self):
        """Setup the characteristic length, time, velocity and pressure of the problem, in order to nondimensionalize the governing equations. The scales are chosen as the following:
        
        Length   L -- bubble radius R
        Velocity U -- the capillary velocity defined as sigma / mu
        Time     T -- L / U
        Pressure P -- sigma / R"""
        R = self.params.physical.R
        mu = self.params.physical.mu
        sigma = self.params.physical.sigma
        V = sigma / mu
        scales = {
            "length": (R, "m"),
            "velocity": (V, "m/s"),
            "time": (R/V, "s"),
            "pressure": (sigma/R, "N/m^2"),
            "acceleration": (V**2/R, "m/s^2")
        }
        self.units = Units(scales)

    def _setup_mesh(self):
        """Setup a NxN square mesh. """
        rm = self.params.control.rm
        N = self.params.control.N
        x = np.linspace(-rm, rm, num=N)
        z = np.linspace(-rm, rm, num=N)
        X, Z = np.meshgrid(x, z) 
        X = X.flatten()
        Z = Z.flatten()
        self.mesh = np.column_stack([X, np.zeros_like(X), Z])
        self.dx = x[1] - x[0] # for computing gradient operators

    def _setup_index_masks(self):
        """Meanwhile get a few index masks (center, edge, center slice) for monitoring the simulation progress."""
        N = self.params.control.N
        mask = np.zeros((N, N)).astype(bool)
        center_ind, edge_ind, centerslice_ind = mask.copy(), mask.copy(), mask.copy()
        center_ind[N//2, N//2] = True
        self.center_ind = center_ind.flatten()
        edge_ind[:, 0], edge_ind[:, -1], edge_ind[0, :], edge_ind[-1, :] = True, True, True, True
        self.edge_ind = edge_ind.flatten()
        centerslice_ind[N//2, :] = True
        self.centerslice_ind = centerslice_ind.flatten()

    def _initial_condition(self):
        H0_dim = self.params.initial.H0
        H0 = self.units.to_nondim(H0_dim, "length")
        h0 = H0 + (self.mesh[:, 0]**2 + self.mesh[:, 2]**2) / 2
        V0_dim = self.params.initial.V0
        V0 = self.units.to_nondim(V0_dim, "velocity")
        theta_rad = self.params.physical.theta / 180 * np.pi
        V0v = np.array([-V0*np.sin(theta_rad), V0*np.cos(theta_rad), 0])
        self.initial_state = np.concatenate([h0, V0v])

    def _setup_gradient_operators(self):
        """Setup gradient operators as sparse matrices."""
        from scipy.sparse import diags, kron, identity
        N = self.params.control.N
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
        R = self.params.physical.R
        rho = self.params.physical.rho
        mu = self.params.physical.mu
        g = self.params.physical.g
        theta_rad = self.params.physical.theta / 180 * np.pi
        gv = np.array([-g*np.sin(theta_rad), g*np.cos(theta_rad), 0])

        # load current state
        h, V = y[:-3], y[-3:]
        h_dim = self.units.to_dim(h, "length")
        V_dim = self.units.to_dim(V, "velocity")

        def _compute_drag(R, V, rho, mu):
            lamb = R * 1.0e3
            chi = (1 - 1.17 * lamb + 2.74 * lamb**2) / (0.74 + 0.45 * lamb)
            s = np.arccos(1 / chi)
            G = 1/3 * chi**(4/3) * (chi**2 - 1)**(3/2) * ((chi**2 - 1)**0.5 - (2-chi**2) * s) / (chi**2 * s - (chi**2 - 1)**0.5)**2
            K = 0.0195 * chi**4 - 0.2134 * chi**3 + 1.7026 * chi**2 - 2.1461 * chi - 1.5732

            Re = 2 * R * rho * np.linalg.norm(V, 2) / mu
            if Re == 0:
                drag = np.array([0, 0, 0])
            else:
                drag = - 48 * G * (1 + K / Re**0.5) * np.pi / 4 * mu * R * V
            return drag
        
        def _compute_amf(H, R, rho, V):
            """Compute added mass force term 2 and added mass coefficient Cm."""
            zeta = (H + R) / R
            print(H, R, zeta)
            Cm = 0.5 + 0.19222 * zeta**-3.019 + 0.06214 * zeta**-8.331 + 0.0348 * zeta**-24.65 + 0.0139 * zeta**-120.7
            dCmdH = (-3.019*0.19222 * zeta**-4.019 - 8.331*0.06214 * zeta**-9.331 - 24.65*0.0348 * zeta**-25.65 - 120.7*0.0139 * zeta**-121.7) / R
            amf2 = -2/3 * np.pi * R**3 * rho * dCmdH * V[1] * V
            return amf2, Cm
        
        def _compute_tff(h):
            """Compute thin film forces."""
            p = self.YL_equation(h)
            p_dim = self.units.to_dim(p, "pressure")
            dhdx = self.Grad["x"] @ h
            dx_dim = self.units.to_dim(self.dx, "length")
            # thin film force x-component
            tffx = - np.sum(p_dim*dhdx) * dx_dim**2
            tffy = np.sum(p_dim) * dx_dim**2
            tff = np.array([tffx, tffy, 0])
            return tff
        
        amf2, Cm = _compute_amf(h_dim[self.center_ind], R, rho, V_dim)

        return {
            "buoyancy": -4/3 * np.pi * R**3 * rho * gv,
            "drag": _compute_drag(R, V_dim, rho, mu),
            "amf2": amf2,
            "tff": _compute_tff(h),
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
        rm = self.params.control.rm
        N = self.params.control.N
        h = y[:-3]
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
            
            where capillary number Ca is defined as 
            
            Ca = mu * V / sigma 
            
            where V is the velocity scale, chosen as sigma / mu, and d, d2 denotes the gradient operator to the first and second order. 

            The boundary condition of dhdt is determined by a force balance equation, which gives the rate of change of bubble velocity. Basically,

            m * dVdt = sum(forces)

            The resulting V[1] is the boundary condition of dhdt at each time step.
            """
            h, V = y[:-3], y[-3:]
            Gx, Gz = self.Grad["x"], self.Grad["z"]

            p = self.YL_equation(h)

            dhdt = (
                V[0] * Gx @ h
                + 1/3 * Gx @ ((Gx @ p) * h**3)
                + 1/3 * Gz @ ((Gz @ p) * h**3)
            )

            dhdt[self.edge_ind] = V[1]

            forces = self.compute_forces(t, y)

            R_dim = self.params.physical.R
            rho = self.params.physical.rho
            dVdt_dim = (forces["buoyancy"] + forces["drag"] + forces["amf2"] + forces["tff"]) / (4/3 * np.pi * R_dim**3 * rho * forces["Cm"])
            dVdt = self.units.to_nondim(dVdt_dim, "acceleration")

            return np.concatenate([dhdt, dVdt])
        
        def event_print(t, y):
            if t - self.last_print >= self.print_interval:
                h, V = y[:-3], y[-3:]
                t_dim = self.units.to_dim(t, "time")
                h_dim = self.units.to_dim(h, "length")
                # V_dim = self.units.to_dim(V, "velocity")
                print(f"t={t_dim*1e3:.2f} ms | hmin={h_dim.min()*1e3:.2f} mm")
                self.last_print = t
                self.update_vis_data(t, y)
                self.update_canvas()
            return 1

        T = self.units.to_nondim(self.params.control.T, "time")

        sol = integrate.solve_ivp(film_drainage, [0, T], self.initial_state, method="BDF", events=event_print, atol=1e-6, rtol=1e-3)

    def run(self, **new_params):
        self.pre_run(**new_params)
        self._run()
        self.post_run()

if __name__ == "__main__":
    sim = BounceSimulator("~/Documents/test", exist_ok=True)
    sim.run()
