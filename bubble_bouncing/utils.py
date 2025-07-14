from bubble_bouncing.bubble import SimulationParams
from dataclasses import fields

def _decomp(y):
    """Decompose the state variable y into h, V and x."""
    return y[:-6], y[-6:-3], y[-3:]

def gradient_operators(N, dx):
    """Construct first and second order gradient operators on a regular square mesh with N vertices on each edge and distance dx between nearest vertices. The operator L takes form of an (N^2, N^2) matrix, which acts on a flattened (N^2, ) spatial variable h to give its derivative:

    L @ h = dh

    where dh is the derivative of h in space. 
    
    Parameters
    ----------
    N : int
        number of vertices on each edge
    dx : float
        distance between nearest vertices

    Returns
    -------
    L : dict[ ndarray[float] ]
        first and second order derivatives in x and z (access by keys "x", "2x", "z" and "2z")

    Note
    ----
    The dimension of the operators is determined by the dimension of dx. For example, if dx is in meters, then the dimension of the first derivative "x" would be meter^-1 and the second derivative "2x" would be meter^-2. Most convenient way is to use dimensionless dx.
    """

    from scipy.sparse import diags, kron, identity

    # 1D first derivative operator (2nd order accuracy)
    Dx = diags([-1, 1], [-1, 1], shape=(N, N))
    Dx = Dx.toarray()
    Dx[0, :3] = [-3, 4, -1]
    Dx[-1, -3:] = [1, -4, 3]
    Dx = Dx / (2*dx)

    # 1D second derivative operator (2nd order accuracy)
    D2x = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
    D2x = D2x.toarray()
    D2x[0, :4] = [2, -5, 4, -1]
    D2x[-1, -4:] = [-1, 4, -5, 2]
    D2x = D2x / (dx)**2

    # 1st and 2nd order derivative operators
    eye = identity(N)
    Gx = kron(eye, Dx)
    Gz = kron(Dx, eye)
    G2x = kron(eye, D2x)
    G2z = kron(D2x, eye)

    return {"x": Gx, "z": Gz, "2x": G2x, "2z": G2z}

def available_keys():
    """Return a list of keys that can be set from command line."""
    paramsd = SimulationParams()
    keys = [f.name for f in fields(paramsd)]
    return keys

def parse_params(unparsed_params):
    """Parse parameters as key-value pairs from command line. Command line command look like
    
    python simulate.py folder --arg1 ARG1 --arg2 ARG2 ...
    
    This function parse the arguments into a dictionary, with argument name as the parameter name, and convert values to correct types according to the SimulationParams dataclass.

    Parameters
    ----------
    unparsed_params : list
        list of unknown arguments to be parsed, typically returned by `parser.parse_known_args()`

    Returns
    -------
    args_dict : dict
        a dict of parsed parameters
    """

    paramsd = SimulationParams()
    keys = [f.name for f in fields(paramsd)]

    # Parse dynamic key-value pairs from unknown args
    it = iter(unparsed_params)
    arg_dict = {}
    for arg in it:
        if arg.startswith('--'):
            key = arg[2:]
            if key in keys:
                try:
                    value = type(getattr(paramsd, key))(next(it))
                except StopIteration:
                    raise ValueError(f"Missing value for argument: --{key}")
                arg_dict[key] = value
    return arg_dict