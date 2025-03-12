"""Zero-dimensional engine cycle ODE system."""

import numpy as np
import cantera as ct
from typing import Tuple, List
from engine_types import InitialConditions, GeometryParameters, OperatingConditions
from area1 import area1
from heat import heat
from fuel import mfuel

def cycle_ode(t: float, y: np.ndarray, gas: ct.Solution, init: InitialConditions, 
              geom: GeometryParameters, oper: OperatingConditions) -> np.ndarray:
    """
    ODE system for a zero-dimensional engine cycle simulation.
    
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
        Cantera Solution object for thermodynamic properties and kinetics
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
    Y_sp = y[4:]  # Species mass fractions [-]
    
    # Specific volume
    vol = V/M  # [m³/kg]
    
    # Update gas state
    gas.TPY = T, P, Y_sp
    
    # Get gas properties
    R_bulk = gas.cp_mass - gas.cv_mass  # [J/(kg·K)]
    CV_bulk = gas.cv_mass  # [J/(kg·K)]
    U_bulk = gas.int_energy_mass  # [J/kg]
    
    # Mass balance - fuel injection
    dmdt, h_f, Y_f = mfuel(t, oper, gas)  # [kg/s], [J/kg], [-]
    
    # Species equations
    # Get species net production rates from Cantera [kg/m³/s]
    y_dot = gas.net_production_rates * gas.molecular_weights / gas.density
    dydt[4:] = dmdt/M * (Y_f - Y_sp) + y_dot
    
    # Volume change equation
    vdt = vdot1(t, init, geom, oper)  # [m³/s]
    dydt[1] = vdt
    
    # Energy equation
    # Calculate gas constants for each species
    R_k = ct.gas_constant / gas.molecular_weights  # [J/(kg·K)]
    sum_RYdot = np.dot(R_k, dydt[4:])  # [J/(kg·K·s)]
    sum_RY = np.dot(R_k, Y_sp)  # [J/(kg·K)]
    
    # Heat transfer
    A = area1(t, init, geom, oper)  # [m²]
    # q, _ = heat(t, V, A, init, geom, oper, gas)  # [W]
    q = 0.0  # Heat transfer disabled for now
    
    # Species energy generation
    h_RT = gas.standard_enthalpies_RT  # [(J/mol)/RT]
    u_nsp = (h_RT * T - T) * R_k  # [J/kg]
    sum_gen = M * np.dot(dydt[4:], u_nsp)  # [W]
    
    # Temperature derivative
    dydt[0] = (1/(M*CV_bulk)) * (-sum_gen - dmdt*U_bulk + dmdt*h_f - P*vdt - q)  # [K/s]
    
    # Pressure derivative
    dydt[2] = P * ((sum_RYdot/sum_RY) - (vdt/V) + (dydt[0]/T) + (dmdt/M))  # [Pa/s]
    
    # Mass derivative
    dydt[3] = dmdt  # [kg/s]
    
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
    # Calculate current crank angle
    theta = init.theta + oper.rpm * 2 * np.pi / 60 * t  # [rad]
    
    # Calculate piston position
    x = geom.a * np.cos(theta) + np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2)
    
    # Calculate volume change rate
    dx_dt = -geom.a * np.sin(theta) * (1 + 
            geom.a * np.cos(theta) / np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2)
            ) * oper.rpm * 2 * np.pi / 60
    
    # Volume change rate is piston area times piston velocity
    return np.pi * (geom.bore/2)**2 * dx_dt  # [m³/s] 