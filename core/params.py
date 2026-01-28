from dataclasses import dataclass

@dataclass(frozen=True)
class HorizonParams:
    W: float
    H: float
    num_horizons: int
    nx: int
    min_thickness: float
    max_thickness: float
    deformation_amplitude: float