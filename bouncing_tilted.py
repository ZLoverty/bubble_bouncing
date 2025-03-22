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
Mar 15, 2025: (i) Use sparse matrix to store gradient operators, this enables efficient large matrix computations; (ii) Use second order accuracy for boundaries; (iii) avoid most reshapes and use flattened 1D arrays in most computations to speed up the computation; (iv) set atol at 1e-6 and rtol at 1e-3 for the balance between speed and convergence. 
Mar 16, 2025: (i) Use only one solve_ivp function to speed up the computation; (ii) Print total simulation time. 
Mar 17, 2025: (i) Non-dimensionalize the equations. Now in most functions, the arguments (t, state) are dimensionless, [film_drainage, YL_equation, save_state, log_force, event_print]; (ii) Use (mu * VT / R) as the scale of the pressure; (iii) Use (R / VT) as time scale.
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
from scipy.sparse import diags, kron, identity
import shutil

def film_drainage(t, state):
    """
    Compute the time derivative of the film thickness h and the bubble velocity V. The film thickness is represented by a 2D array h, and the bubble velocity is represented by a 1D array V. The film thickness `h` and bubble velocity `V` are reshaped into a 1D array `state`. The film thickness is solved by the coupled equations of Stokes-Reynolds equation and Young-Laplace equation, and the bubble velocity is solved by the force balance equation. 
    """
    # break the state into h and V
    h = state[:-3]
    V = state[-3:]
    
    # compute pressure field
    p = YL_equation(h)
    
    # Stokes-Reynolds equation
    dhdy = Gy @ h
    dpdy = Gy @ p
    d2pdy = G2y @ p
    dhdx = Gx @ h
    dpdx = Gx @ p
    d2pdx = G2x @ p
    dhdt = lub_coef1 * (V[0]*dhdx + V[1]*dhdy) + \
        lub_coef2 * h**2 * (h*d2pdx + 3*dhdx*dpdx + h*d2pdy + 3*dhdy*dpdy)

    dhdt[edge_ind] = V[2]

    # force balance model to obtain next velocity dVdt
    buoyancy, drag, amf2, tff, Cm, sound = compute_force(t, state)
    dVdt = (buoyancy + drag + amf2 + tff + sound) / (inertia_coef * Cm) 
    dVdt /= VT / T_scale

    return np.append(dhdt, dVdt)

def YL_equation(h): 

    p = sigma / R / P0 * (2 - L2D @ h)

    p[edge_ind] = 0

    return p


def compute_force(t, state):
    # X, Y = np.meshgrid(x, y)
    # dx = (X[1:-1, 2:  ] - X[1:-1,  :-2]) / 2

    h = state[:-3]
    V = state[-3:] * VT
    t *= T_scale

    H = h[mid_ind] * R
    
    buoyancy = buo

    Re = 2 * R * rho * np.linalg.norm(V, 2) / mu
    if Re == 0:
        drag = np.array([0, 0, 0])
    else:
        drag = - 48 * G * (1 + K / Re**0.5) * np.pi / 4 * mu * R * V

    zeta = (H + R) / R
    Cm = 0.5 + 0.19222 * zeta**-3.019 + 0.06214 * zeta**-8.331 + 0.0348 * zeta**-24.65 + 0.0139 * zeta**-120.7
    # dCmdH = (-3.019*0.19222 * zeta**-4.019 - 8.331*0.06214 * zeta**-9.331 - 24.65*0.0348 * zeta**-25.65 - 120.7*0.0139 * zeta**-121.7) / R
    # amf2 = 2/3 * np.pi * R**3 * rho * dCmdH * V**2

    # Cm = 0.5
    amf2 = np.array([0, 0, 0])

    p = YL_equation(h) * P0
    dhdx = Gx @ h

    # thin film force x-component
    tffx = - np.sum(p*dhdx) * dx**2
    # print(f"p={np.mean(p):.2e}")
    # thin film force z-component
    tffz = np.sum(p) * dx**2

    tff = np.array([tffx, 0, tffz])

    sound = -1 * ampl * np.sin(w * t * 2 * np.pi)
    Sv = [sound * np.cos(theta * np.pi / 180), 0, sound * np.sin(theta * np.pi / 180)]

    return buoyancy, drag, amf2, tff, Cm, Sv

def save_state(t, y):
    """ Save surface shape h and velocity V data to files. The folder structure resembles that of OpenFOAM, where each time step is saved in a separate folder.  """
    t *= T_scale
    h, V = y[:-3] * R, y[-3:] * VT
    save_subfolder = os.path.join(save_folder, f"{t:.5f}")
    os.makedirs(save_subfolder, exist_ok=True)
    np.savetxt(os.path.join(save_subfolder, "h.txt"), h.reshape(N, N))
    np.savetxt(os.path.join(save_subfolder, "V.txt"),  np.array((V,)))

def load_state(load_folder):
    """ Load h and V data from the largest time step. """
    sfL = next(os.walk(load_folder))[1]
    t_current = max(float(sf) for sf in sfL)
    h_file = os.path.join(load_folder, f"{t_current:.5f}", "h.txt")
    V_file = os.path.join(load_folder, f"{t_current:.5f}", "V.txt")

    h = np.loadtxt(h_file) if os.path.exists(h_file) else None
    V = np.loadtxt(V_file) if os.path.exists(V_file) else None

    return t_current, h, V

def log_force(t, y):
    """ need to modify for multi-dimensions """
    force_file = os.path.join(save_folder, "forces.txt")
    buoyancy, drag, amf2, tff, cm, sound = compute_force(t, y)
    t *= T_scale
    h, V = y[:-3] * R, y[-3:] * VT
    H = h[mid_ind]
    if os.path.exists(force_file) == False:
        with open(force_file, "w") as f:
            f.write("{0:>12s}{1:>12s}{2:>12s}{3:>12s}{4:>12s}{5:>12s}{6:>12s}{7:>12s}{8:>12s}{9:>12s}{10:>12s}{11:>12s}{12:>12s}{13:>12s}{14:>12s}{15:>12s}{16:>12s}{17:>12s}{18:>12s}{19:>12s}\n".format("Time", "Distance", "Velocity_x", "Velocity_y", "Velocity_z", "Buoyancy_x", "Buoyancy_y", "Buoyancy_z", "Drag_x", "Drag_y", "Drag_z", "AMF2_x", "AMF2_y", "AMF2_z", "TFF_x", "TFF_y", "TFF_z", "Sound_x", "Sound_y", "Sound_z"))
    with open(force_file, "a") as f:
        f.write("{0:12.8f}{1:12.8f}{2:12.8f}{3:12.8f}{4:12.8f}{5:12.8f}{6:12.8f}{7:12.8f}{8:12.8f}{9:12.8f}{10:12.8f}{11:12.8f}{12:12.8f}{13:12.8f}{14:12.8f}{15:12.8f}{16:12.8f}\n".format(t, H, *V, *buoyancy, *drag, *amf2, *tff, *sound))

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

def precomputes():
    """ Precompute coefficients, constants and operators. """
    # non-dimensionalization coefs
    global lub_coef1, lub_coef2, P0, VT, T_scale
    VT = 2 * R**2 * rho * g / 9 / mu
    T_scale = R / VT
    P0 = mu * VT / R
    lub_coef1 = VT * T_scale / R
    lub_coef2 = P0 * T_scale / mu / 3

    # constants
    global inertia_coef, buo, G, K
    inertia_coef = 4 / 3 * np.pi * rho * R**3
    buo = -4/3 * np.pi * R**3 * rho * gv

    lamb = R * 1.0e3 
    chi = (1 - 1.17 * lamb + 2.74 * lamb**2) / (0.74 + 0.45 * lamb)
    s = np.arccos(1 / chi)
    G = 1/3 * chi**(4/3) * (chi**2 - 1)**(3/2) * ((chi**2 - 1)**0.5 - (2-chi**2) * s) / (chi**2 * s - (chi**2 - 1)**0.5)**2
    K = 0.0195 * chi**4 - 0.2134 * chi**3 + 1.7026 * chi**2 - 2.1461 * chi - 1.5732

    # Define finite difference gradient operators (sparse matrix)
    global Gx, Gy, G2x, G2y, L2D
    # 1D first derivative operator (2nd order accuracy)
    Dx = diags([-1, 1], [-1, 1], shape=(N, N))
    Dx = Dx.toarray()
    Dx[0, :3] = [-3, 4, -1]
    Dx[-1, -3:] = [1, -4, 3]
    Dx = Dx / (2*dx/R)
    # 1D second derivative operator (2nd order accuracy)
    D2x = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
    D2x = D2x.toarray()
    D2x[0, :4] = [2, -5, 4, -1]
    D2x[-1, -4:] = [-1, 4, -5, 2]
    D2x = D2x / (dx/R)**2
    # 1st and 2nd order derivative operators
    eye = identity(N)
    Gx = kron(eye, Dx)
    Gy = kron(Dx, eye)
    G2x = kron(eye, D2x)
    G2y = kron(D2x, eye)
    # 2D Laplacian operator
    L2D = G2x + G2y

    # read sound force magnitude
    global ampl
    url = "https://drive.google.com/uc?export=download&id=1n30z7pKo8teNuiyGZgB47zYMqdac278L"
    response = requests.get(url, stream = True)
    ampls = pd.read_csv(BytesIO(response.content)).set_index("freq")
    ampl = ampls.loc[w, "force"]

def event_print(t, y):
    global last_print_time  # Declare it as global
    h, V = y[:-3], y[-3:]
    # print(f"t={t:f} ms | last_print_time={last_print_time:f} s | print_interval={print_interval:f} s")
    if t - last_print_time >= print_interval:
        last_print_time = t
        print(f"{time.asctime()} -> t={t*T_scale*1e3:.1f} ms | H={h[mid_ind]*R*1e6:.1f} um | Vz={V[2]*VT*1e3:.1f} mm/s")
        save_state(t, y)
        log_force(t, y)
        # test_message(y)
    return 1  # Always return non-zero to avoid terminating

def test_message(y):
    h, V = y[:-3], y[-3:]
    p = YL_equation(h)
    print(f"h_mean={np.mean(h):.2f} | V={np.linalg.norm(V, 2):.2f} | p_mean={np.mean(p)*P0:.2f}")

def main(args):
    # define constants as globals, to avoid passing too many arguments
    ########################################################################################
    global dx, N, R, mu, gv, sigma, rho, w, theta, edge_ind, mid_ind, last_print_time, print_interval, save_folder, g
    last_print_time = 0.0
    ########################################################################################
    
    save_folder = args.save_folder
    load_folder = args.load_folder

    # check if the save folder exists, if yes, remove it
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)

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
        dx = x[1] - x[0]

        # initial state
        t_current = 0 
        h = H0 + (X**2 + Y**2) / R / 2
        V = initial_params["V0"]
        theta = initial_params["theta"]
        Vv = np.array([-V*np.sin(theta/180*np.pi), 0, V*np.cos(theta/180*np.pi)])

        # create an array to mark edge indices
        edge_ind = np.zeros_like(h, dtype=bool)
        edge_ind[:, 0], edge_ind[:, -1], edge_ind[0, :], edge_ind[-1, :] = True, True, True, True
        edge_ind = edge_ind.flatten()

        # find midpoint index
        mid_ind = (N+1)*(N//2)        
        
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
    
    gv = np.array([-g*np.sin(theta/180*np.pi), 0, g*np.cos(theta/180*np.pi)])

    precomputes()

    # compute simulation time steps and save time steps
    T /= T_scale
    t_current /= T_scale
    save_time /= T_scale  
    nSave = np.floor((T-t_current) / save_time).astype("int")
    t_eval = np.linspace(t_current, T, num=nSave)
    print_interval = save_time
    state = np.append(h / R, Vv / VT)

    # prints start message
    print(f"Bubble R={R*1e3:.2f} mm is released !")
    sim_message = f"Total time: {T*T_scale*1e3:.2f} ms | Save time: {save_time*T_scale*1e3:.2f} ms | Time steps to save: {nSave}"
    start_message = "Simulation begins at {}".format(time.asctime())
    print("\n".join([sim_message, start_message, "="*len(start_message)]))


    # log the initial state
    save_state(t_current, state)
    log_initial_params(args)

    # print the initial state
    print(f"{time.asctime()} -> t={t_current*1e3:.1f} ms | H={h[X.shape[0]//2, X.shape[1]//2]*1e6:.1f} um | Vz={Vv[2]*1e3:.1f} mm/s")
    
    t1 = time.time()

    sol = integrate.solve_ivp(film_drainage, [t_current, T], state, \
        t_eval=t_eval,  atol=1e-12, rtol=1e-6, method="LSODA", events=event_print)
    
    t2 = time.time()

    print(f"Simulation time: {t2 - t1:.1f} s")
    # Save data at the last time step
    if sol.success:
        # force remove all the subfolders (since we will generate a more evenly spaced time step)
        sfL = next(os.walk(save_folder))[1]
        for sf in sfL:
            sf_path = os.path.join(save_folder, sf)
            if os.path.exists(sf_path):                
                shutil.rmtree(sf_path)
        os.remove(os.path.join(save_folder, "forces.txt"))
        for i in range(len(sol.t)):
            t = sol.t[i]
            state = sol.y[:, i]
            save_state(t, state)
            log_force(t, state)
    else:
        print(f"Error: {sol.message}")
        print("Simulation failed.")
        

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