"""
bouncing_tilted.py
==================

Description
-----------

This script simulates the evolution of the liquid film between a horizontal solid substrate and an air bubble. This reproduces the simulation in Esmaili 2019. 

Syntax
------

python bouncing_simulation.py save_folder [optional arguments]

save_folder: the folder to save all outputs

optional arguments:
# initial state
    -H, --H0 : initial separation (m), default to 1e-3
    -V, --V0 : initial velocity (m/s), default to -0.3
    -R, --radius : bubble radius (m), default to 6e-4
    -A, --angle : tilted angle of the solid surface (deg)

# simulation parameters
    -T, --time : total time of the simulation (s), default to 0.02
    -N, --number : number of spatial discretization, default to 100
    -s, --save_time : Save the states every save_time (s), default to 1e-4
    --rm : the range of film force integration, the fraction of bubble radius R, default to 0.9
    --load_folder: Folder to load initial state from, default to None.

# physical consts

    --mu : viscosity, Pa s, default to 1e-3
    --g : gravitational acceleration, m/s^2, default to 9.8
    --sigma : surface tension, N/m, default to 72e-3
    --rho : density of water, kg/m^3, default to 997
    --freq : sound frequency in Hz, default to 0


Edit
----

Jul 25, 2024: Initial commit. 
Aug 12, 2024: (i) Add the ability to output information during the simulation; (ii) Add the ability to load initial state from a folder. (iii) Force garbage collection after each time step.
Mar 13, 2025: (i) If `args.freq` is 0, the script will skip the data reading. This allows testing the code without the data file; (ii) Fix the derivative order, y is row and x is column; (iii) Add thin film force in x.
Mar 14, 2025: (i) Add an error handler for the solver: if no solution is found, print the error message and break the loop; (ii) Improve print messages to show the time, height and velocity information; (iii) Use numpy.gradient function to do derivatives, instead of writing out slicing explicitly; (iv) Use global variables to avoid passing too many arguments.
"""

import sys
import os
import numpy as np
from scipy import integrate
import argparse
import json
import time
import pdb
import gc
import pandas as pd


def film_drainage(t, state):
    """
    Compute the time derivative of the film thickness h and the bubble velocity V. The film thickness is represented by a 2D array h, and the bubble velocity is represented by a 1D array V. The film thickness `h` and bubble velocity `V` are reshaped into a 1D array `state`. The film thickness is solved by the coupled equations of Stokes-Reynolds equation and Young-Laplace equation, and the bubble velocity is solved by the force balance equation. 
    """
    # X, Y = np.meshgrid(x, y)

    # break the state into h and V
    h = state[:-3].reshape(dx.shape)
    V = state[-3:]

    # compute differential spaces
    # dy = np.gradient(Y, axis=0, edge_order=2)
    # dx = np.gradient(X, axis=1, edge_order=2)
    
    p = YL_equation(h)

    # velocity boundary conditions
    dhdt = np.zeros_like(h)
    
    # Stokes-Reynolds equation
    dhdy = np.gradient(h, axis=0, edge_order=2) / dy 
    dpdy = np.gradient(p, axis=0, edge_order=2) / dy
    d2pdy = np.gradient(np.gradient(p, axis=0, edge_order=2), axis=0, edge_order=2) / dy**2 
    dhdx = np.gradient(h, axis=1, edge_order=2) / dx 
    dpdx = np.gradient(p, axis=1, edge_order=2) / dx
    d2pdx = np.gradient(np.gradient(p, axis=1, edge_order=2), axis=1, edge_order=2) / dx**2
    dhdt = V[0]*dhdx + V[1]*dhdy + \
        h**2/3/mu * (h*d2pdx + 3*dhdx*dpdx) + \
            h**2/3/mu * (h*d2pdy + 3*dhdy*dpdy)

    dhdt[ : , 0 ] = V[2]
    dhdt[ : ,-1 ] = V[2]
    dhdt[ 0 , : ] = V[2]
    dhdt[-1 , : ] = V[2]

    # pdb.set_trace()
    # print(t, p.shape, dhdx.shape)
    # compute force balance
    buoyancy, drag, amf2, tff, Cm, sound = compute_force(state, t, p, dhdx)
    dVdt = (buoyancy + drag + amf2 + tff + sound) / (4 / 3 * np.pi * rho * R**3 * Cm)
    # print(V[2])

    return np.append(dhdt.reshape(len(x)*len(y)), dVdt)

def YL_equation(h):
    # X, Y = np.meshgrid(x, y)

    # compute differential spaces
    # dy = np.gradient(Y, axis=0, edge_order=2)
    # dx = np.gradient(X, axis=1, edge_order=2)

    d2hdx = np.gradient(np.gradient(h, axis=1, edge_order=2), axis=1, edge_order=2) / dx**2
    d2hdy = np.gradient(np.gradient(h, axis=0, edge_order=2), axis=0, edge_order=2) / dy**2

    p = 2 * sigma / R - sigma * d2hdx - sigma * d2hdy

    p[ : , 0 ] = 0 # p[ : , 1 ]
    p[ : ,-1 ] = 0 # p[ : , -2 ]
    p[ 0 , : ] = 0 # p[ 1 , : ]
    p[-1 , : ] = 0 # p[-2 , : ]

    return p


def compute_force(state, t, p, dhdx):
    # X, Y = np.meshgrid(x, y)
    # dx = (X[1:-1, 2:  ] - X[1:-1,  :-2]) / 2

    h = state[:-3].reshape(dx.shape)
    V = state[-3:]

    H = h[h.shape[0]//2, h.shape[1]//2]
    gv = np.array([-g*np.sin(theta/180*np.pi), 0, g*np.cos(theta/180*np.pi)])

    buoyancy = -4/3 * np.pi * R**3 * rho * gv

    lamb = R * 1.0e3 
    chi = (1 - 1.17 * lamb + 2.74 * lamb**2) / (0.74 + 0.45 * lamb)
    s = np.arccos(1 / chi)
    Re = 2 * R * rho * np.linalg.norm(V, 2) / mu
    G = 1/3 * chi**(4/3) * (chi**2 - 1)**(3/2) * ((chi**2 - 1)**0.5 - (2-chi**2) * s) / (chi**2 * s - (chi**2 - 1)**0.5)**2
    K = 0.0195 * chi**4 - 0.2134 * chi**3 + 1.7026 * chi**2 - 2.1461 * chi - 1.5732
    if Re == 0:
        drag = np.array([0, 0, 0])
    else:
        drag = - 48 * G * (1 + K / Re**0.5) * np.pi / 4 * mu * R * V

    # zeta = (H + R) / R
    # Cm = 0.5 + 0.19222 * zeta**-3.019 + 0.06214 * zeta**-8.331 + 0.0348 * zeta**-24.65 + 0.0139 * zeta**-120.7
    # dCmdH = (-3.019*0.19222 * zeta**-4.019 - 8.331*0.06214 * zeta**-9.331 - 24.65*0.0348 * zeta**-25.65 - 120.7*0.0139 * zeta**-121.7) / R
    # amf2 = - 2/3 * np.pi * R**3 * rho * dCmdH * V**2

    Cm = 0.5
    amf2 = np.array([0, 0, 0])

    # thin film force x-component
    # p = YL_equation(h, x, y, sigma, R)
    # dhdx = (h[1:-1, 2:] - h[1:-1, :-2]) / dx / 2
    # pdhdx = np.zeros_like(p)
    # pdhdx[1:-1, 1:-1] = p[1:-1, 1:-1] * dhdx
    # pdb.set_trace()
    # tffx_tmp = integrate.trapezoid(p*dhdx, x=x, axis=0)
    tffx = 0# integrate.trapezoid(tffx_tmp, x=y, axis=0)

    # thin film force z-component
    tffz_tmp = integrate.trapezoid(p, x=x, axis=0)
    tffz = integrate.trapezoid(tffz_tmp, x=y, axis=0)

    tff = np.array([tffx, 0, tffz])

    def amplitude(w):
        if w == 0:
            return 0
        ampls = pd.read_csv(r"C:\Users\Justi\OneDrive\Cornell\Research.DrJung.Zhengyang\Bubble_cleaning\force_data.csv")
        return ampls.set_index("freq").loc[w, "force"]
    
    sound = -1 * amplitude(w) * np.sin(w * t * 2 * np.pi)
    Sv = [sound * np.cos(theta * np.pi / 180), 0, sound * np.sin(theta * np.pi / 180)]

    return buoyancy, drag, amf2, tff, Cm, Sv

def save_state(save_folder, t_current, h, V):
    """ Save surface shape h and velocity V data to files. The folder structure resembles that of OpenFOAM, where each time step is saved in a separate folder.  """
    save_subfolder = os.path.join(save_folder, f"{t_current:.6f}")
    os.makedirs(save_subfolder, exist_ok=True)
    np.savetxt(os.path.join(save_subfolder, "h.txt"), h)
    np.savetxt(os.path.join(save_subfolder, "V.txt"), np.array((V,)))

def load_state(load_folder):
    """ Load h and V data from the largest time step. """
    sfL = next(os.walk(load_folder))[1]
    t_current = max(float(sf) for sf in sfL)
    h_file = os.path.join(load_folder, f"{t_current:.6f}", "h.txt")
    V_file = os.path.join(load_folder, f"{t_current:.6f}", "V.txt")

    h = np.loadtxt(h_file) if os.path.exists(h_file) else None
    V = np.loadtxt(V_file) if os.path.exists(V_file) else None

    return t_current, h, V

def log_force(save_folder, forces, V, H, t_current):
    """ need to modify for multi-dimensions """
    force_file = os.path.join(save_folder, "forces.txt")
    if os.path.exists(force_file) == False:
        with open(force_file, "w") as f:
            f.write("{0:>12s}{1:>12s}{2:>12s}{3:>12s}{4:>12s}{5:>12s}{6:>12s}{7:>12s}{8:>12s}{9:>12s}{10:>12s}{11:>12s}{12:>12s}{13:>12s}{14:>12s}{15:>12s}{16:>12s}\n".format("Time", "Distance", "Velocity_x", "Velocity_y", "Velocity_z", "Buoyancy_x", "Buoyancy_y", "Buoyancy_z", "Drag_x", "Drag_y", "Drag_z", "AMF2_x", "AMF2_y", "AMF2_z", "TFF_x", "TFF_y", "TFF_z"))
    with open(force_file, "a") as f:
        buoyancy, drag, amf2, tff, cm, sound = forces
        f.write("{0:12.8f}{1:12.8f}{2:12.8f}{3:12.8f}{4:12.8f}{5:12.8f}{6:12.8f}{7:12.8f}{8:12.8f}{9:12.8f}{10:12.8f}{11:12.8f}{12:12.8f}{13:12.8f}{14:12.8f}{15:12.8f}{16:12.8f}\n".format(t_current, H, *V, *buoyancy, *drag, *amf2, *tff))

def log_initial_params(args):
    save_folder = args.save_folder
    initial_params = args_to_dict(args)
    param_file = os.path.join(save_folder, "initial_params.json")
    with open(param_file, "w") as f:
        y = json.dumps(initial_params)
        f.write(y)

def load_initial_params(load_folder):
    param_file = os.path.join(load_folder, "initial_params.json")
    with open(param_file, "r") as f:
        a = f.read()
        initial_params = json.loads(a)
    return initial_params

def args_to_dict(args):
    initial_params = {
        # initial conditions
        "R" : args.radius,
        "H0" : args.H0,
        "V0" : args.V0,
        "theta" : args.angle,

        # physical constant
        "mu" : args.mu,
        "g" : args.g,
        "sigma" : args.sigma,
        "rho" : args.rho,
    
        # simulation params
        "rm" : args.rm,
        "time" : args.time,
        "save_time" : args.save_time,
        "N" : args.number,
        "freq" : args.freq
    }
    return initial_params

#@profile
def main(args):
    global x, y, dx, dy, R, mu, g, sigma, rho, w, theta
    save_folder = args.save_folder
    load_folder = args.load_folder

    if load_folder is None:
        # initialize 
        initial_params = args_to_dict(args)
        rm = initial_params["rm"]
        R = initial_params["R"]
        N = initial_params["N"]
        H0 = initial_params["H0"]
        rm = rm * R
        x = np.linspace(-rm, rm, num=N)
        y = np.linspace(-rm, rm, num=N)

        X, Y = np.meshgrid(x, y)
        dx = np.gradient(X, axis=1, edge_order=2)
        dy = np.gradient(Y, axis=0, edge_order=2) 
        # initial state
        t_current = 0 
        h = H0 + (X**2 + Y**2) / R / 2
        V = initial_params["V0"]
        theta = initial_params["theta"]

        Vv = np.array([-V*np.sin(theta/180*np.pi), 0, V*np.cos(theta/180*np.pi)])

        # save initial state to file
        save_state(save_folder, t_current, h, V)
        log_initial_params(args)
    else:
        # load initial params
        initial_params = load_initial_params(load_folder)
        rm = initial_params["rm"]
        R = initial_params["R"]
        N = initial_params["N"]
        theta = initial_params["theta"]
        
        # load current state t, h, V
        t_current, h, Vv = load_state(load_folder)

    T = initial_params["time"]
    save_time = initial_params["save_time"]
    mu = initial_params["mu"]
    g = initial_params["g"]
    sigma = initial_params["sigma"]
    rho = initial_params["rho"]
    w = initial_params["freq"]

    
    # Summarize initial conditions
    print(f"Bubble R={R*1e3:.2f} mm is released at H0={H0*1e6:.2f} um with V0={Vv} m/s")
    # prints start message
    print("Simulation begins at {}".format(time.asctime()))
    nSave = np.floor((T-t_current) / save_time).astype("int")
    t_eval = np.linspace(t_current, T, num=nSave)
    state = np.append(h, Vv)
    # print the first state
    print(f"{time.asctime()} -> t={t_current*1e3:.1f} ms | H={h[X.shape[0]//2, X.shape[1]//2]*1e6:.1f} um | Vz={Vv[2]*1e3:.1f} mm/s")
    

    # break the integration into small steps
    for i in range(nSave-1):
        t_previous = t_eval[i]
        t_current = t_eval[i+1]
        # pdb.set_trace()
        sol = integrate.solve_ivp(film_drainage, [t_previous, t_current], state, \
            t_eval=[t_current], atol=1e-6, rtol=1e-3, method="BDF")

        # solver error handler
        try:
            state = sol.y[:, -1]
        except:
            print(f"Error in integration: {sol.message}")
            break

        h, Vv = state[:-3].reshape(X.shape), state[-3:]
        save_state(save_folder, t_current, h, Vv)
        p = YL_equation(h)
        dhdx = np.gradient(h, axis=1, edge_order=2) / dx 
        forces = compute_force(state, sol.t[-1], p, dhdx)
        log_force(save_folder, forces, Vv, h[X.shape[0]//2, X.shape[1]//2], t_current)
        print(f"{time.asctime()} -> t={t_current*1e3:.1f} ms | H={h[X.shape[0]//2, X.shape[1]//2]*1e6:.1f} um | Vz={Vv[2]*1e3:.1f} mm/s")
        # release memory after saving the state
        del sol
        gc.collect()

    # t2 = time.monotonic()
    # for i in range(len(sol.t)):
    #     h, Vv = sol.y[:-3, i], sol.y[-3:, i]
    #     save_state(save_folder, sol.t[i], h, Vv)
    #     forces = compute_force(sol.y[:, i], x, y, R=R, rho=rho, g=g, mu=mu, sigma=sigma, theta=theta)
    #     log_force(save_folder, forces, Vv, h[len(h)//2], sol.t[i])
    # t3 = time.monotonic()
    # print("End at {}".format(time.asctime()))
    # print("Solving equations takes {:f} seconds".format(t2-t1))
    # print("Saving data takes {:f} seconds".format(t3-t2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="bouncing_simulation", description="Simulate the evolution of the liquid film between a solid substrate and an air bubble.")
    parser.add_argument("save_folder", help="Folder to save simulation results.")
    # initial state
    parser.add_argument("-H", "--H0", type=float, default=1.0e-3, help="Initial separation in meters.")
    parser.add_argument("-V", "--V0", type=float, default=-0.3, help="Initial velocity in m/s.")
    parser.add_argument("-R", "--radius", type=float, default=6.0e-4, help="Bubble radius in meters.")
    parser.add_argument("-A", "--angle", type=float, default=22.5, help="Tilted angle of solid surface.")

    # simulation params
    parser.add_argument("-T", "--time", type=float, default=0.02, help="Total simulation time in seconds.")
    parser.add_argument("-N", "--number", type=int, default=50, help="Number of discretization points.")
    parser.add_argument("-s", "--save_time", type=float, default=1e-4, help="Save the states every save_time.")
    parser.add_argument("--rm", type=float, default=0.9, help="Range of integration, when times R.")
    parser.add_argument("--load_folder", type=str, default=None, help="Folder to load initial state from.")
    
    # physical params
    parser.add_argument("--mu", type=float, default=1e-3, help="Viscosity in Pa s.")
    parser.add_argument("--g", type=float, default=9.8, help="Gravitational acceleration in m/s^2.")
    parser.add_argument("--sigma", type=float, default=72e-3, help="Surface tension in N/m.")
    parser.add_argument("--rho", type=float, default=997, help="Density of water in kg/m^3.")
    parser.add_argument("--freq", type=int, default=0, help="Sound frequency in Hz.")

    args = parser.parse_args()
    main(args)