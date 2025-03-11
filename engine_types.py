"""Common data types for engine simulation."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class OperatingConditions:
    """Engine operating conditions."""
    soc: float  # Start of cycle [rad]
    rpm: float  # Engine RPM
    tin: float  # Inlet temperature [K]
    pin: float  # Inlet pressure [Pa]
    tex: float  # Exhaust temperature [K]
    pex: float  # Exhaust pressure [Pa]
    compin: Dict[str, float]  # Composition inlet
    twall: float  # Wall temperature [K]
    Upbar: float  # Mean piston speed [m/s]

@dataclass
class InitialConditions:
    """Initial engine conditions."""
    theta: float  # Initial crank angle [rad]

@dataclass
class GeometryParameters:
    """Engine geometry parameters."""
    a: float      # Crank radius [m]
    conr: float   # Connecting rod length [m]
    bore: float   # Cylinder bore [m]
    x: Optional[float] = None  # Instantaneous piston position [m]

@dataclass
class ValveData:
    """Valve geometry and flow data."""
    ca: np.ndarray  # Crank angle array
    lift: np.ndarray  # Lift array
    refd: float  # Reference diameter
    ra: np.ndarray  # Reference area array
    cd: np.ndarray  # Discharge coefficient array

class Gas:
    """Gas state and properties."""
    def __init__(self):
        self._pressure = 0.0
        self._temperature = 0.0
        self._density = 0.0
        self._cp = 0.0
        self._cv = 0.0
        self._composition = {}

    def set(self, T: float, P: float, composition: Dict[str, float]) -> None:
        """Set gas properties."""
        self._temperature = T
        self._pressure = P
        self._composition = composition
        # Note: In a real implementation, these would be calculated based on gas properties
        # Here we're using placeholder calculations
        self._density = P / (287.05 * T)  # Using ideal gas law with R = 287.05 J/(kg·K)
        self._cp = 1005.0  # Approximate cp for air at room temperature
        self._cv = 718.0   # Approximate cv for air at room temperature

    def pressure(self) -> float:
        """Get gas pressure [Pa]."""
        return self._pressure

    def temperature(self) -> float:
        """Get gas temperature [K]."""
        return self._temperature

    def density(self) -> float:
        """Get gas density [kg/m³]."""
        return self._density

    def cp_mass(self) -> float:
        """Get specific heat at constant pressure [J/(kg·K)]."""
        return self._cp

    def cv_mass(self) -> float:
        """Get specific heat at constant volume [J/(kg·K)]."""
        return self._cv 