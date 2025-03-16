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
        Geometry parameters
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
    T = max(y[0], 200.0)  # Prevent temperature from going too low
    V = y[1]  # Volume [m³]
    P = y[2]  # Pressure [Pa]
    M = y[3]  # Total mass [kg]
    
    # Update gas state
    gas.TPY = T, P, y[4:]
    
    # Get gas properties
    cp = gas.cp_mass
    cv = gas.cv_mass
    R = ct.gas_constant / gas.mean_molecular_weight
    
    # Volume change equation
    vdt = vdot1(t, init, geom, oper)  # [m³/s]
    dydt[1] = vdt
    
    # Calculate chemical reaction rates
    wdot = gas.net_production_rates  # [kmol/m³/s]
    mdot = wdot * gas.molecular_weights  # [kg/m³/s]
    
    # Species mass fraction rates [1/s]
    ydot = mdot / gas.density
    dydt[4:] = ydot
    
    # Calculate instantaneous cylinder area
    A = area1(t, init, geom, oper)  # [m²]
    
    # Calculate heat transfer using Woschni correlation
    q, _ = heat(t, V, A, init, geom, oper, gas)  # [W]
    
    # Energy equation including chemical heat release and heat transfer
    # dT/dt = -1/(M*cv) * (P*dV/dt - Q_chem + Q_wall)
    Q_chem = -np.sum(gas.partial_molar_enthalpies * wdot) * V  # [W]
    dydt[0] = -1.0/(M*cv) * (P*vdt - Q_chem + q)
    
    # Pressure equation from ideal gas law
    # P = rho * R * T
    dydt[2] = P * (dydt[0]/T - vdt/V)
    
    # Mass is constant in closed system
    dydt[3] = 0.0
    
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