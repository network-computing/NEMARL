from dataclasses import dataclass


@dataclass
class WaveletConfig:
    mechanism: str = "exact"
    heat_coefficient: float = 1000.0
    sample_number: int = 16
    approximation: int = 100
    step_size: int = 20
    switch: int = 100

