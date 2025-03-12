"""Common data types for engine simulation."""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import cantera as ct

@dataclass
class OperatingConditions:
    """Engine operating conditions."""
    # Basic parameters
    soc: float  # Start of cycle [rad]
    rpm: float  # Engine RPM
    tin: float  # Inlet temperature [K]
    pin: float  # Inlet pressure [Pa]
    tex: float  # Exhaust temperature [K]
    pex: float  # Exhaust pressure [Pa]
    compin: Dict[str, float]  # Composition inlet
    twall: float  # Wall temperature [K]
    Upbar: float  # Mean piston speed [m/s]
    R: float = ct.gas_constant  # Universal gas constant [J/(mol·K)]
    
    # Valve timing
    evc: float  # Exhaust valve closing [deg]
    ivo: float  # Intake valve opening [deg]
    ivc: float  # Intake valve closing [deg]
    evo: float  # Exhaust valve opening [deg]
    nvlv: int = 2  # Number of valves
    
    # Intake and exhaust properties
    Yin: Optional[np.ndarray] = None  # Intake composition [-]
    Hin: Optional[float] = None  # Intake enthalpy [J/kg]
    Yex: Optional[np.ndarray] = None  # Exhaust composition [-]
    Hex: Optional[float] = None  # Exhaust enthalpy [J/kg]

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
    ca: np.ndarray  # Crank angle array [deg]
    lift: np.ndarray  # Lift array [mm]
    refd: float  # Reference diameter [mm]
    ra: np.ndarray  # Reference area array [-]
    cd: np.ndarray  # Discharge coefficient array [-]
    dur: Optional[float] = None  # Valve duration [deg]

class Gas:
    """Gas state and properties."""
    def __init__(self):
        self._pressure = 0.0
        self._temperature = 0.0
        self._density = 0.0
        self._cp = 0.0
        self._cv = 0.0
        self._composition = {}
        self._species_list = []
        self._molar_masses = {}  # [kg/mol]
        self._enthalpies = {}    # [J/kg]

    def set(self, T: float, P: float, composition: Dict[str, float]) -> None:
        """Set gas properties."""
        self._temperature = T
        self._pressure = P
        self._composition = composition
        self._species_list = list(composition.keys())
        
        # Note: In a real implementation, these would be calculated based on gas properties
        # Here we're using placeholder calculations for air
        self._density = P / (287.05 * T)  # Using ideal gas law with R = 287.05 J/(kg·K)
        self._cp = 1005.0  # Approximate cp for air at room temperature
        self._cv = 718.0   # Approximate cv for air at room temperature
        
        # Placeholder values for species properties
        for species in self._species_list:
            self._molar_masses[species] = 0.029  # Approximate for air [kg/mol]
            self._enthalpies[species] = self._cp * T  # Simplified enthalpy [J/kg]

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

    def internal_energy(self) -> float:
        """Get specific internal energy [J/kg]."""
        return self._cv * self._temperature

    def molar_masses(self) -> np.ndarray:
        """Get molar masses for all species [kg/mol]."""
        return np.array([self._molar_masses[s] for s in self._species_list])

    def enthalpies_RT(self) -> np.ndarray:
        """Get dimensionless enthalpies (h/RT) for all species."""
        RT = 8.314 * self._temperature  # [J/mol]
        return np.array([self._enthalpies[s]/RT for s in self._species_list])

    def composition_array(self) -> np.ndarray:
        """Get composition as numpy array in species order."""
        return np.array([self._composition[s] for s in self._species_list]) 