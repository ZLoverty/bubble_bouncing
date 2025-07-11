"""
Define simulation parameters and their default values. The Simulator object will be initiated with these parameters, and a `set_params` method is provided to update custom parameters for specific simulation cases.
"""
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Type, TypeVar


@dataclass
class PhysicalConstants:
    g: float = 9.8
    mu: float = 1e-3
    sigma: float = 72e-3
    rho: float = 1e3
    R: float = 6e-4
    theta: float = 22.5

@dataclass
class InitialConditions:
    V0: float = -0.3
    H0: float = 1e-3

@dataclass
class SimulationControl:
    T: float = 0.2
    rm: float = 1.2
    N: int = 100
    save_time: float = 1e-4

@dataclass
class SimulationParams:
    physical: PhysicalConstants = field(default_factory=PhysicalConstants)
    initial: InitialConditions = field(default_factory=InitialConditions)
    control: SimulationControl = field(default_factory=SimulationControl)

T = TypeVar("T")

def from_dict(cls: Type[T], d: dict) -> T:
    """Recursively construct a dataclass from a nested dictionary. This is useful when we need to read parameters from a .yml config file."""
    
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name)
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)

def update_params(params, **updates):
    """Recursively set fields in `params` with the updated values in the `updates` dictionary."""
    for f in fields(params):
        val = getattr(params, f.name)
        if is_dataclass(val):
            update_params(val, **updates)
        else:
            if f.name in updates:
                setattr(params, f.name, updates[f.name])
    return params