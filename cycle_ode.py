"""Zero-dimensional engine cycle ODE system."""

import numpy as np
import cantera as ct
from typing import Tuple, List
from engine_types import InitialConditions, GeometryParameters, OperatingConditions
from area1 import area1
from heat import heat
from mfuel import mfuel

def cycle_ode(t: float, y: np.ndarray, gas: ct.Solution, init: InitialConditions, 
              geom: GeometryParameters, oper: OperatingConditions) -> np.ndarray:
    """
    Main ODE system for ideal gas compression/expansion.
    
    Parameters
    ----------
    t : float
        Time [s]
    y : np.ndarray
        Solution vector with components:
            y[0] = Temperature [K]
            y[1] = Volume [m³]
            y[2] = Pressure [Pa]
            y[3] = Total mass [kg]
            y[4:] = Species mass fractions [-]
    gas : ct.Solution
        Cantera Solution object for thermodynamic properties
    init : InitialConditions
        Initial conditions
    geom : GeometryParameters
        Engine geometry parameters
    oper : OperatingConditions
        Operating conditions
    
    Returns
    -------
    np.ndarray
        Time derivatives of state variables [dy/dt]
    """
    # Initialize derivative vector
    dydt = np.zeros_like(y)
    
    # Extract state variables
    T = y[0]  # Temperature [K]
    V = y[1]  # Volume [m³]
    P = y[2]  # Pressure [Pa]
    M = y[3]  # Total mass [kg]
    
    # Update gas state
    gas.TPY = T, P, y[4:]
    
    # Get gas properties
    gamma = gas.cp_mass / gas.cv_mass  # Ratio of specific heats
    
    # Volume change equation
    vdt = vdot1(t, init, geom, oper)  # [m³/s]
    dydt[1] = vdt
    
    # For adiabatic compression/expansion of ideal gas:
    # PV^γ = constant
    # dP/dt = -γ * P * dV/dt / V
    dydt[2] = -gamma * P * vdt / V  # [Pa/s]
    
    # For ideal gas, T = PV/(MR)
    # dT/dt = (V*dP/dt + P*dV/dt)/(MR)
    R = ct.gas_constant / gas.mean_molecular_weight
    dydt[0] = (V * dydt[2] + P * vdt) / (M * R)  # [K/s]
    
    # Mass is constant
    dydt[3] = 0.0
    
    # Species mass fractions are constant (for now)
    dydt[4:] = 0.0
    
    return dydt

def vdot1(t: float, init: InitialConditions, geom: GeometryParameters, 
          oper: OperatingConditions) -> float:
    """
    Calculate rate of volume change.
    
    Parameters
    ----------
    t : float
        Time [s]
    init : InitialConditions
        Initial conditions
    geom : GeometryParameters
        Engine geometry parameters
    oper : OperatingConditions
        Operating conditions
    
    Returns
    -------
    float
        Rate of volume change [m³/s]
    """
    # Use current crank angle directly from oper.soc
    # At TDC (0°), piston is at top with only clearance volume
    theta = oper.soc * np.pi/180  # Convert to radians, 0° at TDC
    
    # Calculate piston position from TDC
    x = geom.a * np.cos(theta) + np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2)
    
    # Calculate volume change rate using RPM directly
    # Note: dx_dt is positive when piston moves down (volume increases)
    dx_dt = -geom.a * np.sin(theta) * (1 + 
            geom.a * np.cos(theta) / np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2)
            ) * oper.rpm * 2 * np.pi / 60
    
    # Volume change rate is piston area times piston velocity
    # During compression (EVC to IVO), volume should decrease
    return -np.pi * (geom.bore/2)**2 * dx_dt  # [m³/s] 