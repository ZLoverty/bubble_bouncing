from dataclasses import dataclass

@dataclass
class SimulationParams:
    """Dataclass to hold the default bubble simulation parameters."""
    # Physical constants:
    g: float = 9.8
    mu: float = 1e-3
    sigma: float = 72e-3
    rho: float = 1e3
    R: float = 6e-4
    theta: float = 22.5
    lift_coef: float = 1.0

    # Initial conditions:
    V0: float = -0.3
    H0: float = 1e-3

    # Simulation control:
    T: float = 0.2
    rm: float = 1.2
    N: int = 100
    save_time: float = 1e-3
    print_time: float = 1e-5