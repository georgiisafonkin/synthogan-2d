from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class HorizonParams:
    W: float
    H: float
    num_horizons: int
    nx: int
    min_thickness: float
    max_thickness: float
    deformation_amplitude: float
    seed: Optional[int] = None
    min_gap: Optional[float] = None