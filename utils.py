import numpy as np

def clip_data(vec_data, max_mag):
    """Clip a vector data to set an upper bound to the magnitude. The vector data is of shape (npts, dims), where magnitude is defined as norm(axis=1)"""
    mag = np.linalg.norm(vec_data, axis=1)
    scale = np.minimum(1.0, max_mag / (mag + 1e-12))  # Avoid divide-by-zero
    clipped = vec_data * scale[:, np.newaxis]
    return clipped

class Sphere:
    """Compute the flow field in a Oseen wake. The flow field is characterized by two regions: a Stokeslet in the low Reynolds region and a compensating flow """
    def __init__(self, a, U=0, rho=1e3, mu=1e-3):
        self.a = a # bubble radius
        self.U = U # upward velocity (only upward!)
        self.rho = rho # density
        self.mu = mu # viscosity
        self.pos = np.array([0, 0, 0]) # bubble position
        self.surf_coords, self.unit_normals, self.ds = self._compute_surface_coords()
        self.unit_tangents = self._compute_surface_tangent_xz()

    def Oseen_wake(self, points):
        """Compute Oseen wake at given points. 

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
        r = x**2 + y**2 + z**2
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