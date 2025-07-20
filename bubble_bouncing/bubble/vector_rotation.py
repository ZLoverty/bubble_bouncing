import numpy as np

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