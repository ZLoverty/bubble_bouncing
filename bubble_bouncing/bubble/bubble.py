import numpy as np
import healpy as hp
from typing import Sequence

class Bubble:
    """Compute the forces and the flow field associated with a moving bubble in a liquid. The forces we consider are buoyancy, drag, added mass force and thin film force when the bubble touches a solid surface. The flow field we consider is an Oseen wake. The flow field is characterized by two regions: a Stokeslet in the low Reynolds region and a compensating flow."""

    def __init__(self, 
                 a: float, 
                 U: Sequence[float] = (1,0,0), 
                 rho: float = 1e3, 
                 mu: float = 1e-3):
        self.a = a # bubble radius
        self.U = np.array(U) # upward velocity (only upward!)
        self.rho = rho # density
        self.mu = mu # viscosity
        self.pos = np.array([0, 0, 0]) # bubble position
        self.surf_coords, self.unit_normals, self.ds = self._compute_surface_coords()
        self.unit_tangents = self._compute_surface_tangent_xy()

    def Oseen_wake(self, points):
        """Compute Oseen wake at given (relative) points. 

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
        """
        U = self.U
        a = self.a
        rho = self.rho
        mu = self.mu
        U_mag = np.linalg.norm(U)
        Re = rho * a * U_mag / mu

        # Rotate the points to a reference frame where U points to positive z direction
        rel_pos = points - self.pos
        rel_pos_z = self._rotate_to_z(rel_pos)

        x, y, z = rel_pos_z[:, 0], rel_pos_z[:, 1], rel_pos_z[:, 2]
        r = (x**2 + y**2 + z**2) ** 0.5
        sint = (x**2 + y**2)**0.5 / r
        cost = z / r
        sinp = y / (x**2 + y**2)**0.5
        cosp = x / (x**2 + y**2)**0.5

        u_r = U_mag * (
            - a**3 * cost / 2 / r**3 
            + 3 * a**2 / (2 * r**2 * Re) * (
                1 - np.exp(- r * Re / 2 / a * (1 + cost))
            )
            - 3 * a * (1 - cost) / 4 / r * np.exp(- r * Re / 2 / a * (1 + cost))
        )
        u_t = U_mag * (
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
        flow_ = np.stack([u_x, u_y, u_z], axis=-1)

        # rotate the velocity field back to the original frame
        flow = self._rotate_from_z(flow_)

        return flow
    
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

    def _compute_rotation_axis_and_angle(self, U_unit, zn):
        """Compute the rotation axis and angle to rotate the flow field from z direction to U_unit direction. This is a helper function to perform coordinate transformation.
        
        Parameters
        ----------
        U_unit : array_like
            A 3D vector representing the direction of the flow field.
        zn : array_like
            A 3D vector representing the positive z direction, usually [0, 0, 1].

        Returns
        -------
        k : ndarray
            The axis of rotation, a 3D vector.
        alpha : float
            The angle in radians by which to rotate the vector.
        """
        k = np.cross(zn, U_unit)
        k /= np.linalg.norm(k)
        cos_alpha = np.dot(zn, U_unit)
        alpha = np.arccos(cos_alpha)
        return k, alpha
    
    def _rotate_to_z(self, pos):
        """Rotate the points to a reference frame where U points to positive z direction. This is a helper function to perform coordinate transformation.
        
        Parameters
        ----------
        pos : array_like
            An array of shape (N, 3), the points to be rotated to the frame where U aligns with +z.

        Returns
        -------
        pos_rotated : Vector
            The rotated points as a Vector object.
        """
        U_unit = self.U / np.linalg.norm(self.U)
        zn = np.array([0, 0, 1])
        k, alpha = self._compute_rotation_axis_and_angle(U_unit, zn)
        pos = Vector(pos)
        pos_rotated = pos.rotate(k, alpha)
        return pos_rotated
    
    def _rotate_from_z(self, pos):
        """Rotate the points to a reference frame where U points to positive z direction. This is a helper function to perform coordinate transformation.
        
        Parameters
        ----------
        pos : array_like
            An array of shape (N, 3), the points to be rotated to the problem frame.

        Returns
        -------
        pos_rotated : Vector
            The rotated points as a Vector object.
        """
        U_unit = self.U / np.linalg.norm(self.U)
        zn = np.array([0, 0, 1])
        k, alpha = self._compute_rotation_axis_and_angle(U_unit, zn)
        pos = Vector(pos)
        pos_rotated = pos.rotate(k, -alpha)
        return pos_rotated
    
def _skew_symmetric(k):
    """
    Convert a vector k to a skew-symmetric matrix.
    
    Parameters
    ----------
    k : array_like
        A 3D vector.
    
    Returns
    -------
    K : ndarray
        A 3x3 skew-symmetric matrix corresponding to the vector k.
    """

    if np.isclose(np.linalg.norm(k), 1.0) == False:
        k = k / np.linalg.norm(k)
        
    K = np.array([[0.   , -k[2], k[1] ],
                    [k[2] , 0.   , -k[0]],
                    [-k[1], k[0] , 0.   ]])
    return K

def rotation_matrix_axis(k, angle):
    """
    Rotate a vector v around an axis k by a given angle. This function computes the rotation matrix using the Rodrigues' rotation formula.
    
    Parameters
    ----------
    v : array_like
        A 3D vector to be rotated.
    k : array_like
        A 3D vector representing the axis of rotation.
    angle : float
        The angle in radians by which to rotate the vector.
    
    Returns
    -------
    v_rotated : ndarray
        The rotated vector.
    """

    if np.isclose(np.linalg.norm(k), 0.0):
        return np.eye(3)
    else:
        k = k / np.linalg.norm(k)
    
    K = _skew_symmetric(k)
    
    R = (
        np.eye(3) * np.cos(angle) 
        + K * np.sin(angle) 
        + np.outer(k, k) * (1 - np.cos(angle))
    )
    
    return R

class Vector(np.ndarray):
    """
    A subclass of numpy.ndarray to represent 3D vector(s).
    
    This class allows for easy rotation of the vector(s).
    """
    
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.shape[1] != 3:
            raise ValueError("Vector must be a 3D vector.")
        return obj
    
    def rotate(self, k, angle):
        """
        Rotate the vector around an axis k by a given angle.
        
        Parameters
        ----------
        k : array_like
            A 3D vector representing the axis of rotation.
        angle : float
            The angle in radians by which to rotate the vector.
        
        Returns
        -------
        Vector
            The rotated vector.

        Example
        -------
        >>> points = np.random.rand(10, 3)
        >>> k = np.array([0, 0, 1])
        >>> angle = np.pi / 4  # 45 degrees
        >>> rotated_points = Vector(points).rotate(k, angle)
        """
        R = rotation_matrix_axis(k, angle)
        return Vector((R @ self.T).T)
    
