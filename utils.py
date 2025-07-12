def _decomp(y):
    """Decompose the state variable y into h, V and x."""
    return y[:-6], y[-6:-3], y[-3:]