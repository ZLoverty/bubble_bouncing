"""
A class to define the characteristic scales of the simulation, so that the conversion between dimensionful and dimensionless quantities can be controlled in a centralized manner. 
"""

class Units:
    """Centralized unit conversion control."""
    def __init__(self, scales_dict: dict):
        """Pass a dict of scales. The keys in this dict is used to rescale the quantities."""
        self.scales = scales_dict
        self.available_keys = ", ".join(scales_dict.keys())

    def __repr__(self):
        lines = []
        for key in self.scales:
            value = self.scales[key][0]
            unit = self.scales[key][1]
            line = f"{key} scale: {value} {unit}"
            lines.append(line)
        return "\n".join(lines) 
    
    def to_dim(self, nondim_value, quantity):
        if quantity in self.scales:
            return nondim_value * self.scales[quantity][0]
        else:
            raise ValueError(f"The quantity specified is not available. Please enter the following quantities: {self.available_keys}")
        
    def to_nondim(self, dim_value, quantity):
        if quantity in self.scales:
            return dim_value / self.scales[quantity][0]
        else:
            raise ValueError(f"The quantity specified is not available. Please enter the following quantities: {self.available_keys}")