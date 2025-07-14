"""
forces.py
=========

This module contains the implementation of forces acting on a bubble in a fluid simulation.
"""

import numpy as np

def compute_buoyancy(R, rho, g):
    """Compute the buoyancy force acting on a bubble. The formula used is:

    F_b = -4/3 * pi * R**3 * rho * g,

    where R is the radius of the bubble, rho is the density of the fluid, and g is the gravitational acceleration.

    Parameters:
    -----------
    R : float
        Radius of the bubble.
    rho : float
        Density of the fluid.
    g : ndarray
        Gravitational acceleration vector.

    Returns:
    --------
    buoyancy : ndarray
        Buoyancy force vector.
    """
    return -4/3 * np.pi * R**3 * rho * g

def compute_drag(R, V, rho, mu):
    """Compute the drag force acting on a bubble. The equation used is 
    
    F_D = -48 * G * (1 + K / Re**0.5) * pi / 4 * mu * R * V,

    where G and K are correction factors as functions of the bubble radius.
    
    Parameters:
    ----------
    R : float
        Radius of the bubble.
    V : np.ndarray
        Velocity vector of the bubble.
    rho : float
        Density of the fluid.
    mu : float
        Dynamic viscosity of the fluid.

    Returns:
    -------
    drag : np.ndarray
        Drag force vector.
    """
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

def compute_amf(H, R, rho, V):
    """Compute added mass force term 2 and added mass coefficient Cm. Formula used is:

    amf2 = -2/3 * pi * R**3 * rho * dCmdH * V[1] * V,

    where dCmdH is the derivative of the added mass coefficient Cm with respect to H.

    Cm = 0.5 + 0.19222 * zeta**-3.019 + 0.06214 * zeta**-8.331 + 0.0348 * zeta**-24.65 + 0.0139 * zeta**-120.7,

    where zeta = (H + R) / R.
    
    Parameters:
    -----------
    H : float
        Distance from the center of bubble bottom surface to the solid surface.
    R : float
        Radius of the bubble.
    rho : float
        Density of the fluid.
    V : np.ndarray
        Velocity vector of the bubble.    
        
    Returns:
    --------
    amf2 : np.ndarray
        Added mass force term 2.
    Cm : float
        Added mass coefficient.
    """
    zeta = (H + R) / R
    Cm = 0.5 + 0.19222 * zeta**-3.019 + 0.06214 * zeta**-8.331 + 0.0348 * zeta**-24.65 + 0.0139 * zeta**-120.7
    dCmdH = (-3.019*0.19222 * zeta**-4.019 - 8.331*0.06214 * zeta**-9.331 - 24.65*0.0348 * zeta**-25.65 - 120.7*0.0139 * zeta**-121.7) / R
    amf2 = -2/3 * np.pi * R**3 * rho * dCmdH * V[1] * V
    return amf2, Cm

def compute_tff(p, dhdx, dx):
    """Compute thin film forces by integrating the pressure across the bubble surface. Since the bubble profile is asymmetric in the x-direction, we also need to consider the x-component of the thin film force. The formula used is:
    
    F_x = \int p * dhdx ds
    F_y = \int p ds

    where ds is the differential area element along the bubble surface. In our computation, we approximate ds as dx^2.

    Parameters:
    -----------
    h : np.ndarray
        Height profile of the bubble.
    p : np.ndarray
        Pressure profile of the bubble.
    dhdx : np.ndarray
        Derivative of the height profile with respect to x.
    dx : float
        Spatial step size in the x-direction.

    Returns:
    --------
    tff : np.ndarray
        Thin film force vector.

    Note:
    -----
    All the input arguments of this function (h, p, dhdx, dx) are in dimensionful form. The conversion should be done in the main loop prior to calling this function.
    """
    ds = dx ** 2
    tffx = - np.sum(p*dhdx) * ds
    tffy = np.sum(p) * ds
    tff = np.array([tffx, tffy, 0])
    return tff

def compute_lift(a, surface_flow, ds, U, lift_coef=1.0):
    """Compute the circulation induced by the Oseen wake flow. We use the following formula
    
    Gamma = 1/2a \int_S u_s dS
    lift = 4 * np.pi * a**3 / 3 * lift_coef / np.pi / a**2 * (Gamma x U)

    Parameters
    ----------
    a : float
        bubble radius
    surface_flow : ndarray[float]
        flow field projection on the intersection between bubble surface and xy plane
    ds : float
        differential area on the bubble surface
    U : float
        bubble velocity
    lift_coef : float
        lift coefficient
    
    Return
    ------
    lift : float
        lift force

    Note
    ----
    The definition of surface tangents in the Bubble class dictates that positive Gamma represents CW rotation, where the direction of vorticity points in +z. Therefore, the vector form Gamma should be the magnitude times e_z. 

    Example
    -------
    >>> flow = im.Oseen_wake(re.surf_coords+re.pos)
    >>> surface_flow = flow * re.unit_tangents
    >>> lift = compute_lift(a, surface_flow, re.ds, re.U)
    """
    
    Gamma = 1 / 2 / a * surface_flow.sum() * ds * np.array([0, 0, 1])
    lift = 4 * np.pi * a**3 / 3 * lift_coef / np.pi / a**2 * np.cross(Gamma, U)
    return lift