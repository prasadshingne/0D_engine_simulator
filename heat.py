"""Heat transfer calculations using Woschni correlation."""

from typing import Tuple
import numpy as np
from engine_types import OperatingConditions, InitialConditions, GeometryParameters

def _calculate_woschni_coefficient(t: float, vol: float, pressure: float, temperature: float,
                                init: InitialConditions, geom: GeometryParameters,
                                oper: OperatingConditions) -> float:
    """
    Calculate heat transfer coefficient using Woschni correlation.
    
    Parameters
    ----------
    t : float
        Time [s]
    vol : float
        Volume [m³]
    pressure : float
        Pressure [Pa]
    temperature : float
        Temperature [K]
    init : InitialConditions
        Initial conditions
    geom : GeometryParameters
        Geometry parameters
    oper : OperatingConditions
        Operating conditions
    
    Returns
    -------
    float
        Heat transfer coefficient [W/(m²·K)]
    """
    # Calculate current crank angle
    theta = init.theta + oper.rpm * 2 * np.pi / 60 * t  # [rad]
    
    # Calculate instantaneous piston speed
    dx_dt = -geom.a * np.sin(theta) * (1 + 
            geom.a * np.cos(theta) / np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2)
            ) * oper.rpm * 2 * np.pi / 60  # [m/s]
    
    # Woschni correlation coefficients
    C1 = 2.28  # Coefficient for mean piston speed term
    C2 = 0.00324  # Coefficient for motoring pressure term
    
    # Calculate characteristic velocity
    # During compression/expansion: w = C1*Up
    # During combustion/expansion: w = C1*Up + C2*(Vd*T1)/(p1*V1)*(p-pm)
    w = C1 * abs(dx_dt)  # [m/s]
    
    # Add combustion term if after TDC
    if theta > 0:
        # Get reference conditions at IVC
        V1 = geom.Vcl + np.pi/4 * geom.bore**2 * (geom.conr + geom.a - 
             (geom.a * np.cos(init.theta) + 
              np.sqrt(geom.conr**2 - (geom.a * np.sin(init.theta))**2)))  # [m³]
        p1 = oper.pin  # [Pa]
        T1 = oper.tin  # [K]
        
        # Calculate motoring pressure (assuming polytropic compression)
        gamma = 1.35  # Specific heat ratio for air
        Vm = geom.Vcl + np.pi/4 * geom.bore**2 * (geom.conr + geom.a - 
             (geom.a * np.cos(theta) + 
              np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2)))  # [m³]
        pm = p1 * (V1/Vm)**gamma  # [Pa]
        
        # Add combustion term
        w += C2 * geom.Vd * T1 / (p1 * V1) * (pressure - pm)  # [m/s]
    
    # Constants for Woschni correlation
    C = 3.26  # Overall coefficient
    d = geom.bore  # Characteristic length [m]
    
    # Calculate heat transfer coefficient
    # h = C*d^(-0.2)*p^(0.8)*T^(-0.53)*w^(0.8)
    h = C * d**(-0.2) * (pressure/1e5)**0.8 * temperature**(-0.53) * w**0.8
    
    return h

def heat(t: float, vol: float, area: float, init: InitialConditions,
         geom: GeometryParameters, oper: OperatingConditions, gas: object) -> Tuple[float, float]:
    """
    Calculate heat transfer coefficient and heat loss.
    
    Parameters
    ----------
    t : float
        Time [s]
    vol : float
        Volume [m³]
    area : float
        Area [m²]
    init : InitialConditions
        Initial conditions
    geom : GeometryParameters
        Geometry parameters
    oper : OperatingConditions
        Operating conditions
    gas : object
        Gas object with pressure and temperature properties
    
    Returns
    -------
    Tuple[float, float]
        Heat loss [W], Heat transfer coefficient [W/(m²·K)]
    """
    # Get pressure and temperature
    pressure = gas.P  # [Pa]
    temperature = gas.T  # [K]
    
    # Calculate heat transfer coefficient
    h = _calculate_woschni_coefficient(t, vol, pressure, temperature, init, geom, oper)
    
    # Calculate heat loss (positive = loss to walls)
    q = h * area * (temperature - oper.twall)
    
    return q, h 