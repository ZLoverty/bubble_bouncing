import h5py
from pathlib import Path

class DataBuffer(list):
    def __init__(self, 
                 h5file: str | Path, 
                 dset_name: str, 
                 data_shape: tuple = (),
                 dtype : type = float):
        """Initiate a DataBuffer object. The main purpose is to implement a `flush()` method, so that a list of data can be flushed to a designated file on the disk.
        
        Parameters
        ----------
        h5file : str | Path
            path to the .h5 file, where the data are dumped
        dset_name : str
            dataset name
        data_shape : tuple
            data shape at each step as a tuple, e.g. () for scalar
        dtype : type
            date type

        Examples
        --------
        >>> with h5py.File(h5file, "w") as f: pass
        >>> t_buffer = DataBuffer(h5file, "t")
        >>> for t in range(100):
                t_buffer.append(t)
                if len(t_buffer) > thres:
                    t_buffer.flush()
        """
        super().__init__()
        self.dset_name = dset_name
        self.h5file = Path(h5file)
        try:
            with h5py.File(self.h5file, "r+") as f:
                f.create_dataset(self.dset_name, (0, *data_shape), maxshape=(None, *data_shape), dtype=dtype)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{h5file} not found!")
        
    def flush(self):
        """Flush the buffered data to h5file."""
        n_add = len(self)
        dset_name = self.dset_name

        with h5py.File(self.h5file, "r+") as f:
            dset = f[dset_name]
            dset.resize(dset.shape[0]+n_add, axis=0)
            dset[-n_add:] = self
        self.clear()