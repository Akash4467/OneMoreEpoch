from importlib.metadata import version as _version

from onemoreepoch.core import Device, Parameter, Tensor

__version__ = _version("onemoreepoch")
__all__ = ["Device", "Parameter", "Tensor", "__version__"]
