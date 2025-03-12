"""Calculate fuel injection parameters."""

from typing import Tuple
import cantera as ct
import numpy as np
from engine_types import OperatingConditions

def mfuel(t: float, oper: OperatingConditions, gas: ct.Solution) -> Tuple[float, float, np.ndarray]:
    """
    Calculate fuel injection parameters.
    
    Parameters
    ----------
    t : float
        Current time [s]
    oper : OperatingConditions
        Operating conditions
    gas : ct.Solution
        Cantera Solution object
    
    Returns
    -------
    mf : float
        Fuel mass flow rate [kg/s]
    hf : float
        Fuel specific enthalpy [J/kg]
    yf : np.ndarray
        Fuel mass fractions [-]
    """
    # Calculate current crank angle
    CAD = oper.soc + oper.rpm/60*360*t  # [deg]
    
    # Check if we're in the injection window
    if CAD >= oper.injt and CAD <= oper.injt + oper.injdur:
        # Calculate injection duration in seconds
        dt = oper.injdur * np.pi/180 / (oper.rpm * 2 * np.pi/60)  # [s]
        
        # Get current cylinder conditions
        Pcyl = gas.P  # [Pa]
        Tcyl = gas.T  # [K]
        
        # Set gas state to fuel conditions (200 bar above cylinder pressure)
        fuel_comp = 'C8H18:1'  # Fuel composition
        gas.TPX = Tcyl, Pcyl + 200e5, fuel_comp  # Temperature [K], Pressure [Pa], Composition
        
        # Get fuel properties
        hf = gas.h  # Specific enthalpy [J/kg]
        yf = gas.Y  # Mass fractions [-]
        
        # Calculate mass flow rate (mg/cycle -> kg/s)
        mf = (oper.minj/1e6)/dt  # [kg/s]
    else:
        # No injection
        hf = 0.0
        yf = np.zeros_like(gas.Y)
        mf = 0.0
    
    return mf, hf, yf 