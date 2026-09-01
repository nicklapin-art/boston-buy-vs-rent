"""Boston buy-versus-rent Monte Carlo simulator."""

from .config import SimulationConfig
from .simulation import SimulationResult, run_simulation

__all__ = ["SimulationConfig", "SimulationResult", "run_simulation"]
__version__ = "0.1.0"

