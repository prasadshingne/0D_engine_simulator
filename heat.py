from typing import Tuple
from engine_types import OperatingConditions, Gas

def _calculate_woschni_coefficient(vol: float, pressure_kpa: float, temperature: float, mean_piston_speed: float) -> float:
    """
    Calculate heat transfer coefficient using modified Woschni correlation.
    
    Args:
        vol: Volume [m³]
        pressure_kpa: Pressure [kPa]
        temperature: Temperature [K]
        mean_piston_speed: Mean piston speed [m/s]
    
    Returns:
        Heat transfer coefficient [W/(m²·K)]
    """
    # Constants for modified Woschni correlation
    C1 = 130.0  # Coefficient
    C2 = 4.0    # Volume exponent
    C3 = 0.8    # Pressure exponent
    C4 = -0.4   # Temperature exponent
    C5 = 0.8    # Velocity exponent
    
    return (C1 * vol**C2 * (pressure_kpa)**C3 * 
            temperature**C4 * (mean_piston_speed + 1.4)**C5)

def heat(t: float, vol: float, area: float, init: object, geom: object, 
        oper: OperatingConditions, gas: Gas) -> Tuple[float, float]:
    """
    Calculate heat transfer coefficient and heat loss.
    
    Parameters:
    -----------
    t : float
        Time [s]
    vol : float
        Volume [m³]
    area : float
        Area [m²]
    init : object
        Initial conditions
    geom : object
        Geometry parameters
    oper : OperatingConditions
        Engine operating conditions
    gas : Gas
        Gas state object with pressure and temperature methods
    
    Returns:
    --------
    Tuple[float, float]
        (q, hh) where:
        q : float
            Instantaneous heat loss [W]
        hh : float
            Heat transfer coefficient [W/(m²·K)]
    """
    # Get pressure and temperature from gas object
    pressure_kpa = gas.pressure() / 1e3  # Convert Pa to kPa
    temperature = gas.temperature()
    
    # Calculate heat transfer coefficient using Woschni correlation
    hh = _calculate_woschni_coefficient(
        vol=vol,
        pressure_kpa=pressure_kpa,
        temperature=temperature,
        mean_piston_speed=oper.Upbar
    )
    
    # Calculate instantaneous heat loss
    q = hh * area * (temperature - oper.twall)
    
    return q, hh 