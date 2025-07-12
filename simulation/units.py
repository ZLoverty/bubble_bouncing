class Units:
    """Centralized unit conversion control. A class to define the characteristic scales of the simulation, so that the conversion between dimensionful and dimensionless quantities can be controlled in a centralized manner. """

    def __init__(self, scales_dict: dict):
        """Pass a dict of scales. The keys in this dict is used to rescale the quantities.
        
        Parameters
        ----------
        scales_dict : dict
            A dictionary where keys are the names of the quantities and values are tuples containing the scale and the unit.
            Example: {'length': (1e-6, 'm'), 'time': (1e-3, 's')}"""
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
        """Convert a nondimensional value to a dimensional value using the scale defined in the scales dictionary.
        
        Parameters
        ----------
        nondim_value : float or array_like
            The nondimensional value to be converted
        quantity : str
            The type of the quantity to be converted, which should match a key in the scales dictionary, e.g. "length", "time", etc.

        Returns
        -------
        dim_value : float or array_like
            The dimensional value corresponding to the nondimensional value.
        """
        if quantity in self.scales:
            return nondim_value * self.scales[quantity][0]
        else:
            raise ValueError(f"The quantity specified is not available. Please enter the following quantities: {self.available_keys}")
        
    def to_nondim(self, dim_value, quantity):
        """Convert a dimensional value to a nondimensional value using the scale defined in the scales dictionary.
        
        Parameters
        ----------
        dim_value : float or array_like
            The dimensional value to be converted to nondimensional.
        quantity : str
            The type of the quantity to be converted, which should match a key in the scales dictionary, e.g. "length", "time", etc.

        Returns
        -------
        nondim_value : float or array_like
            The nondimensional value corresponding to the dimensional value.
        """
        if quantity in self.scales:
            return dim_value / self.scales[quantity][0]
        else:
            raise ValueError(f"The quantity specified is not available. Please enter the following quantities: {self.available_keys}")