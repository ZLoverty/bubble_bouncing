"""
bbfem.py
========

Use Finite Element Method to solve the thin film equation. This would hopefully improve the computation speed as well as the convergence of the solver. This script makes use of the FEniCSx package. 

Syntax
------

python bbfem.py [--flags] [--args ARGS]

Edit
----
* Jun 25, 2025: Initial commit.
"""

from petsc4py import PETSc
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from dolfinx import fem, io, default_scalar_type
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from basix.ufl import element, mixed_element
from ufl import (
    FacetNormal, Identity, div, inner, dx, ds,
    transpose, dot, as_vector, outer, lhs, rhs,
    TestFunction, TrialFunction, nabla_grad, derivative, CellDiameter, Measure
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BubbleBouncingSimulator:
    """
    Simulates the evolution of flow and director field in an active nematic system using FEniCSx.
    """
    def __init__(self, args):
        """
        args -- a dictionary of arguments.
        """
        self.args = self.setup_args(args)
        self.save_dir = Path(self.args["save_dir"]).expanduser().resolve()
        self.mesh_dir = Path(self.args["mesh_dir"]).expanduser().resolve()
        self.log_file_path = self.save_dir / "bbfem.log"
        self.setup_directories()
        self._configure_file_logging()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.domain, self.cell_tags, self.facet_tags = self._load_mesh()
        self.ds = Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.wall_measures = sum([self.ds(tag) for tag in self.args["wall_tags"]], start=self.ds(0))
        self.constants = self._define_constants()
        self.function_spaces = self._define_function_spaces()
        self.functions = self._define_functions()
        self.solver_tf = self._setup_thin_film_solver()
        self._initialize_states()
        self._write_simulation_parameters()
        self.precomputes()
        self.force_balance()
        self.compute_average_mesh_size()
        self.writer = io.VTXWriter(self.comm, self.save_dir / "results.pvd",
                                   output=[self.functions["p_n"], self.functions["h_n"]])
    
    def setup_directories(self):
        """Creates the save directory and copies the mesh file."""

        if self.save_dir.exists() and not self.args["f"]:
            print(f"Simulation {self.save_dir} already exists, abort ...")
            exit()
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.mesh_dir, self.save_dir / "mesh.msh")
     
    def setup_args(self, args):
        """overwrite the default values with provided args."""

        # this dict defines the default params, unless arguments are passed through CLI, these params will be used for the simulation.
        default_args = {
            "save_dir": "~/.bbfem",
            "mesh_dir": "mesh.msh",
            "total_time": 10,
            "dt": 1e-4,
            "H0": 1e-3,
            "V0": [0, 0, 0],
            "radius": 6e-4,
            "angle": 22.5,
            "wall_tags": [1],
            "save_time": 1e-4,
            "f": False,
            "mu": 1e-3,
            "g": 9.8,
            "rho": 1e3,
            "sigma": 72e-3,
            "freq": 0
        }
        
        for key in args:
            if key in default_args:
                if key == "V0":
                    V0 = args["V0"]
                    theta = args["angle"]
                    default_args[key] = [-V0*np.sin(theta/180*np.pi), 0, V0*np.cos(theta/180*np.pi)]
                else:
                    default_args[key] = args[key]
        
        return default_args

    def _configure_file_logging(self):
        """Adds a file handler to the logger."""
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    def _load_mesh(self):
        """Loads the mesh from the specified directory."""
        logger.info(f"Loading mesh from {self.mesh_dir}")
        domain, cell_tags, facet_tags = io.gmshio.read_from_msh(self.mesh_dir, self.comm, 0, gdim=2)
        logger.info(f"All facet_tags: {np.unique(facet_tags.values)}")
        return domain, cell_tags, facet_tags

    def _define_constants(self):
        """Defines and returns simulation constants."""

        constants = {
            "H0": fem.Constant(self.domain, PETSc.ScalarType(self.args["H0"])),
            "V0": fem.Constant(self.domain, PETSc.ScalarType(np.array(self.args["V0"]))),
            "radius": fem.Constant(self.domain, PETSc.ScalarType(self.args['radius'])),
            "angle": fem.Constant(self.domain, PETSc.ScalarType(self.args["angle"])),
            "mu": fem.Constant(self.domain, PETSc.ScalarType(self.args["mu"])),
            "g": fem.Constant(self.domain, PETSc.ScalarType(self.args["g"])),
            "sigma": fem.Constant(self.domain, PETSc.ScalarType(self.args["sigma"])),
            "rho": fem.Constant(self.domain, PETSc.ScalarType(self.args["rho"])),
            "freq": fem.Constant(self.domain, PETSc.ScalarType(self.args["freq"])),
            "dt": fem.Constant(self.domain, PETSc.ScalarType(self.args["dt"]))
        }

        g = self.args["g"]
        mu = self.args["mu"]
        R = self.args["radius"]
        rho = self.args["rho"]
        sigma = self.args["sigma"]

        self.l_ref = R
        self.v_ref = rho * g / 9 / mu * R**2
        self.t_ref = R / self.v_ref
        self.p_ref = sigma / R

        return constants

    def _define_function_spaces(self):
        """Defines and returns the function spaces."""
        h_el = element("Lagrange", self.domain.topology.cell_name(), 1, shape=())
        p_el = element("Lagrange", self.domain.topology.cell_name(), 1, shape=())
 
        V_h = fem.functionspace(self.domain, h_el)
        V_p = fem.functionspace(self.domain, p_el)

        return {"V_h": V_h,"V_p": V_p}

    def _define_functions(self):
        """Defines and returns dolfinx functions."""
        V_h = self.function_spaces["V_h"]
        V_p = self.function_spaces["V_p"]
        
        functions = {
            "h_n": fem.Function(V_h),
            "p_n": fem.Function(V_p),
            "h_": fem.Function(V_h),
            "p_": fem.Function(V_p), 
            "h_dim": fem.Function(V_h, name="height"),
            "p_dim": fem.Function(V_p, name="pressure")
        }
        return functions

    def _setup_thin_film_solver(self):
        """Sets up the Navier-Stokes linear problem."""
        
        phi = TestFunction(self.function_spaces["V_p"])
        h_n = self.functions["h_n"]
        h_ = self.functions["h_"]
        p_n = self.functions["p_"]

        sigma = self.constants["sigma"].value
        mu = self.constants["mu"].value
        dt = self.constants["dt"].value / self.t_ref

        V01 = fem.Constant(self.domain, self.constants["V0"].value[:2] / self.v_ref)
        V2 = fem.Constant(self.domain, self.constants["V0"].value[2] / self.v_ref)
                          
        # Thin-film equation
        F1 = (
            (h_ - h_n) / dt * phi * dx
            - inner(V01, nabla_grad(h_)) * phi * dx
            + sigma / 3 / mu / self.v_ref * inner(nabla_grad(p_n*h_**3), nabla_grad(phi)) * dx
            + ((h_ - h_n) / dt - V2) * phi * self.wall_measures
        )

        J = derivative(F1, h_)
        problem = fem.petsc.NonlinearProblem(F1, h_, J=J)
        solver = NewtonSolver(self.comm, problem)

        return solver

    def _initialize_states(self):
        """Initializes the Q-tensor."""
        H0 = self.constants["H0"].value
        self.functions["h_n"].interpolate(lambda x: H0/self.l_ref + (x[0]**2+x[1]**2)/2)
        logger.info("Initial h state set.")

    def _write_simulation_parameters(self):
        """Writes simulation parameters to a JSON file."""
        with open(self.save_dir / "params.json", 'w') as json_file:
            json.dump(self.args, json_file, indent=4)
        logger.info("Simulation parameters written to params.json.")

    def compute_average_mesh_size(self):
        """Compute the average mesh size."""
        self.domain.topology.create_connectivity(1, 0)  # edges to vertices
        edges = self.domain.topology.connectivity(1, 0).array.reshape(-1, 2)
        edge_coords = self.domain.geometry.x[edges]
        edge_lengths = np.linalg.norm(edge_coords[:, 0, :] - edge_coords[:, 1, :], axis=1)
        self.avg_h = np.mean(edge_lengths)
    
    def force_balance(self):
        R = self.args["radius"]
        rho = self.args["rho"]
        g = self.args["g"]
        theta = self.args["angle"]
        gv = np.array([-g*np.sin(theta/180*np.pi), 0, g*np.cos(theta/180*np.pi)])
        
        buoyancy = self.buo

        mu = self.args["mu"]
        V = self.constants["V0"].value
        Re = 2 * R * rho * np.linalg.norm(V, 2) / mu
        if Re == 0:
            drag = np.array([0, 0, 0])
        else:
            drag = - 48 * self.G * (1 + self.K / Re**0.5) * np.pi / 4 * mu * R * V

        H = self.functions["h_n"].x.array.min() * self.l_ref
        zeta = (H + R) / R
        Cm = 0.5 + 0.19222 * zeta**-3.019 + 0.06214 * zeta**-8.331 + 0.0348 * zeta**-24.65 + 0.0139 * zeta**-120.7
        dCmdH = (-3.019*0.19222 * zeta**-4.019 - 8.331*0.06214 * zeta**-9.331 - 24.65*0.0348 * zeta**-25.65 - 120.7*0.0139 * zeta**-121.7) / R
        amf2 = -2/3 * np.pi * R**3 * rho * dCmdH * V[2] * V

        # how to integrate over a mesh? I don't know yet.
        p = self.functions["p_n"]
        h = self.functions["h_n"]

        print(A)

    def precomputes(self):
        """Compute some constant values."""
        R = self.args["radius"]
        rho = self.args["rho"]
        g = self.args["g"]
        theta = self.args["angle"]
        gv = np.array([-g*np.sin(theta/180*np.pi), 0, g*np.cos(theta/180*np.pi)])

        self.buo = -4/3 * np.pi * R**3 * rho * gv

        lamb = R * 1.0e3 
        chi = (1 - 1.17 * lamb + 2.74 * lamb**2) / (0.74 + 0.45 * lamb)
        s = np.arccos(1 / chi)
        self.G = 1/3 * chi**(4/3) * (chi**2 - 1)**(3/2) * ((chi**2 - 1)**0.5 - (2-chi**2) * s) / (chi**2 * s - (chi**2 - 1)**0.5)**2
        self.K = 0.0195 * chi**4 - 0.2134 * chi**3 + 1.7026 * chi**2 - 2.1461 * chi - 1.5732




    def run(self):
        """Executes the main simulation loop."""
        logger.info(f"Simulation starts at {time.asctime()}!")
        t0 = time.time()
        t = 0.0
        step_total = int(self.args["total_time"] / self.args["dt"])
        h_dim = self.functions["h_dim"]
        p_dim = self.functions["p_dim"]
        try:
            for i in range(step_total):
                # Data visualization and output
                t_dim = t * self.t_ref
                h_dim.x.array[:] = self.functions["h_n"].x.array * self.l_ref
                p_dim.x.array[:] = self.functions["p_n"].x.array * self.p_ref
                self.writer.write(t_dim)

                t = (i + 1) * self.args["dt"] / self.t_ref # Update time for next step
      
                if self.rank == 0:
                    hmin = self.functions["h_dim"].x.array.min()
                    logger.info(f"{i+1}/{step_total}, t={t*self.t_ref:.4f}, hmin={hmin:.4f}, T_lapse={time.time()-t0:.0f} s")

                # Solve Q-tensor evolution equation
                num_iterations, converged = self.solver_tf.solve(self.functions["h_"])
                if not converged:
                    logger.warning(f"Q-tensor solver did not converge at step {i+1}. Iterations: {num_iterations}")
                self.functions["h_n"].interpolate(self.functions["h_"])

            if self.rank == 0:
                logger.info(f"COMPLETED: {step_total}/{step_total}, t={t*self.t_ref:.4f}, hmin={hmin:.4f}, T_lapse={time.time()-t0:.0f} s")

        except Exception as e:
            logger.error(f"An unexpected error occurred during simulation at step {i+1}: {e}", exc_info=True)
            if self.rank == 0:
                logger.error(f"EXIT EARLY: {i+1}/{step_total}")
        finally:
            self.writer.close()
            logger.info("Simulation finished. VTX writer closed.")
            # Ensure file handler is closed
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)

def main():
    parser = argparse.ArgumentParser(prog="bouncing_simulation", description="Simulate the evolution of the liquid film between a solid substrate and an air bubble.")

    parser.add_argument("mesh_dir", type=str, help="Folder to save simulation results.")
    parser.add_argument("-o", "--save_dir", type=str, default="~/.bbfem", help="Folder to save simulation results.")

    # initial state
    parser.add_argument("-H", "--H0", type=float, default=1.0e-3, help="Initial separation in meters.")
    parser.add_argument("-V", "--V0", type=float, default=-0.3, help="Initial velocity in m/s.")
    parser.add_argument("-R", "--radius", type=float, default=6.0e-4, help="Bubble radius in meters.")
    parser.add_argument("-A", "--angle", type=float, default=22.5, help="Tilted angle of solid surface.")

    # simulation params
    parser.add_argument("-T", "--total_time", type=float, default=0.2, help="Total simulation time in seconds.")
    parser.add_argument("-s", "--save_time", type=float, default=1e-4, help="Save the states every save_time.")
    parser.add_argument("--rm", type=float, default=0.9, help="Range of integration, when times R.")
    parser.add_argument("--load_folder", type=str, default=None, help="Folder to load initial state from.")
    parser.add_argument("-f", action="store_true", help="Force running the simulation without checking existence.")
    parser.add_argument("--dt", type=float, default=1e-3, help="Total simulation time in seconds.")

    
    # physical params
    parser.add_argument("--mu", type=float, default=1e-3, help="Viscosity in Pa s.")
    parser.add_argument("--g", type=float, default=9.8, help="Gravitational acceleration in m/s^2.")
    parser.add_argument("--sigma", type=float, default=72e-3, help="Surface tension in N/m.")
    parser.add_argument("--rho", type=float, default=997, help="Density of water in kg/m^3.")
    parser.add_argument("--freq", type=int, default=0, help="Sound frequency in Hz.")

    args = parser.parse_args()
    simulator = BubbleBouncingSimulator(vars(args))
    simulator.run()

if __name__ == "__main__":
    main()