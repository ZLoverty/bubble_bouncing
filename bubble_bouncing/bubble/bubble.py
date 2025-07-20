import numpy as np
import healpy as hp
from typing import Sequence
from .vector_rotation import Vector

class Bubble:
    """Compute the forces and the flow field associated with a moving bubble in a liquid. The forces we consider are buoyancy, drag, added mass force and thin film force when the bubble touches a solid surface. The flow field we consider is an Oseen wake. The flow field is characterized by two regions: a Stokeslet in the low Reynolds region and a compensating flow."""

    def __init__(self, 
                 a: float, 
                 U: Sequence[float] = (1., 0., 0.), 
                 rho: float = 1e3, 
                 mu: float = 1e-3):
        self.zn = np.array([0., 0., 1.])
        self.a = a # bubble radius
        self.U = np.array(U) # upward velocity (only upward!)
        self.rho = rho # density
        self.mu = mu # viscosity
        self.pos = np.array([0., 0., 0.]) # bubble position
        self.surf_coords, self.unit_normals, self.ds = self._compute_surface_coords()
        self.unit_tangents = self._compute_surface_tangent_xy()
        

    def Oseen_wake(self, points):
        """Compute Oseen wake at given (relative) points. This function extends the `Oseen_wake_z` function by considering sphere velocity in arbitrary directions. This is done by rotating the points to z first, then compute flow field, and finally rotate the flow field back to the original coordinate system.

        Parameters
        ----------
        points : ndarray
            an array of shape (N, 3), the points to evaluate Oseen wake flow field.

        Returns: 
        -------
        flow : ndarray
            the flow velocity at each given point, also of shape(N, 3).

        Note
        ----
        The points here are the positions in the problem reference frame, i.e. the absolute positions of the problem. For example, if we consider the surface of another bubble as the points, `points = bubble.surf_coords + bubble.pos`. The relative position is handled inside the method.

        Example
        -------
        >>> im = Bubble(a, U=U)
        >>> re = Bubble(a, U=U1)
        >>> flow = im.Oseen_wake(re.surf_coords+re.pos)
        """
        U = self.U
        zn = self.zn
        U_mag = np.linalg.norm(U)

        # Rotate the points to a reference frame where U points to positive z direction
        rel_pos = points - self.pos
        rel_pos_z = self.rotate(rel_pos, U, -zn)

        flow_ = self.Oseen_wake_z(U_mag, rel_pos_z)
        
        # rotate the velocity field back to the original frame
        flow = self.rotate(flow_, -zn, U)

        return flow
    
    def Oseen_wake_z(self, U, points):
        """Compute the Oseen wake flow induced by a sphere moving in +z direction at velocity U. 
        This is a direct implementation of the Oseen wake formula in the book "An introduction to suspension dynamics" by Guazzelli and Morris.

        Parameters
        ----------
        U : float
            Velocity magnitude in +z
        points : ndarray
            Relative positions to evaluate the flow velocities. Mush be of shape (N, 3).
        
        Returns
        -------
        flow : ndarray
            Flow velocity of the Oseen wake. Same shape as points.

        Note
        ----
        Since the original formula assumes velocity in -z direction, we need to rotate the resulting flow field around x or y by pi.
        """

        if points.shape[1] != 3:
            raise ValueError("Points shape must be (N, 3).")
        
        a = self.a
        rho = self.rho
        mu = self.mu
        zn = self.zn
        Re = rho * a * U / mu

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = (x**2 + y**2 + z**2) ** 0.5
        sint = (x**2 + y**2)**0.5 / r
        cost = z / r
        sinp = y / (x**2 + y**2)**0.5
        cosp = x / (x**2 + y**2)**0.5

        u_r = U * (
            - a**3 * cost / 2 / r**3 
            + 3 * a**2 / (2 * r**2 * Re) * (
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
        """Get the position of the bubble."""
        return self.pos
    
    def set_pos(self, pos):
        """Set the position of the bubble."""
        self.pos = np.array(pos)

    def set_velocity(self, U):
        """Update the bubble velocity during the simulation."""
        self.U = U
        
    def _compute_surface_coords(self, nside=4):
        """Compute the coordinates of the surface differential area and surface unit normal vectors.
        
        Parameters
        ----------
        nside : int 
            controls how many parts the spherical surface is to be divided, utilizing the `healpy` package. Has to be power of 2. The number of parts will be 12*nside**2.
        
        Returns
        -------
        surface_coords : np.ndarray
            coordinates of surface differential area __relative to the center__ of the sphere.
        unit_normals : np.ndarray
            unit normal vectors corresponding to the surface locations.
        differential_surface_area : float
            the area of each differential surface unit
        """

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
    
    def _compute_surface_tangent_xy(self):
        """Compute sphere surface tangential unit vectors, specifically their projections on the xy plane."""
        tangent = np.zeros_like(self.unit_normals)
        tangent[:, 0] = self.unit_normals[:, 1]
        tangent[:, 1] = - self.unit_normals[:, 0]
        return tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    
    def _grid(self):
        """Create a grid of points around the bubble for testing flow field."""
        lim = self.a * 3
        N = 10
        x = np.linspace(-lim, lim, N)
        y = np.linspace(-lim, lim, N)
        z = np.linspace(-lim, lim, N)
        x, y, z = np.meshgrid(x, y, z)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        return np.stack([x, y, z], axis=-1)
    
    def rotate(self, vec, from_, to):
        """Rotate a vector field from one coordinates to another. 
        
        Paremeters
        ----------
        vec : ndarray
            Vector field to be rotated. Must be of shape (N, 3).
        from_ : array
            Director of the original coordinates. Must be of shape (3,).
        to : array
            Director of the rotated coordinates. Must be of shape (3,).
        
        Returns
        -------
        rotated : ndarray
            Rotated vector field.
        """
        k = np.cross(from_, to)
        dot = (from_ * to).sum()
        
        k_norm = np.linalg.norm(k)
        if np.isclose(k_norm, 0.0):
            # if from_ and to are in the same or opposite direction, we can use an arbitrary direction that is perpendicular to to.
            k = np.array([-to[2], 0, to[0]])
            k /= np.linalg.norm(k)
            if dot > 0:
                alpha = 0.0
            else:
                alpha = np.pi
        else:
            k /= k_norm
            cos_alpha = np.dot(from_, to)
            alpha = np.arccos(cos_alpha)
 
        vec = Vector(vec)
        rotated = vec.rotate(k, alpha)
        return rotated
    



    
