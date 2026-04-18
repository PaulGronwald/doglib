"""triplot — matplotlib tripartite (shock response spectrum) projection.

Importing the package registers the 'tripartite' projection with matplotlib.
"""
from matplotlib.projections import register_projection

from .axes import TripartiteAxes
from .units import IMPERIAL, SI, UnitSystem

register_projection(TripartiteAxes)

__all__ = ["TripartiteAxes", "UnitSystem", "IMPERIAL", "SI"]
__version__ = "0.1.0"
