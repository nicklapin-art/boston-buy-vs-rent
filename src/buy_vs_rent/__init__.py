"""Boston buy-versus-rent Monte Carlo simulator."""

from .config import SimulationConfig
from .simulation import SimulationResult, run_simulation
from .uncertainty import ParameterUncertaintyResult, run_parameter_uncertainty

__all__ = [
    "ParameterUncertaintyResult",
    "SimulationConfig",
    "SimulationResult",
    "run_parameter_uncertainty",
    "run_simulation",
]
__version__ = "0.2.0"
