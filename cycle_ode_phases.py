"""Zero-dimensional engine cycle ODE system with different phases."""

import numpy as np
import cantera as ct
from typing import Tuple, List, Optional
from engine_types import InitialConditions, GeometryParameters, OperatingConditions, ValveData
from area1 import area1
from heat import heat
from valve_flow import mair_in, mchout

def _common_ode_calc(t: float, y: np.ndarray, gas: ct.Solution, init: InitialConditions,
                     geom: GeometryParameters, oper: OperatingConditions,
                     dmdt: float, h_in_out: float, Y_in_out: np.ndarray) -> np.ndarray:
    """
    Common ODE calculations shared between all cycle phases.
    
    Parameters
    ----------
    t : float
        Time [s]
    y : np.ndarray
        Solution vector
    gas : ct.Solution
        Cantera Solution object
    init : InitialConditions
        Initial conditions
    geom : GeometryParameters
        Engine geometry parameters
    oper : OperatingConditions
        Operating conditions
    dmdt : float
        Mass flow rate [kg/s]
    h_in_out : float
        Enthalpy of incoming/outgoing flow [J/kg]
    Y_in_out : np.ndarray
        Mass fractions of incoming/outgoing flow [-]
    
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
    
    # Update gas state
    gas.TPY = T, P, Y_sp
    
    # Get gas properties
    CV_bulk = gas.cv_mass  # [J/(kg·K)]
    U_bulk = gas.int_energy_mass  # [J/kg]
    H_bulk = gas.enthalpy_mass  # [J/kg]
    
    # Species equations
    # Get species net production rates from Cantera [kg/m³/s]
    y_dot = gas.net_production_rates * gas.molecular_weights / gas.density
    dydt[4:] = dmdt/M * (Y_in_out - Y_sp) + y_dot
    
    # Volume change equation
    vdt = vdot1(t, init, geom, oper)  # [m³/s]
    dydt[1] = vdt
    
    # Energy equation
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
    dydt[0] = (1/(M*CV_bulk)) * (-sum_gen - dmdt*U_bulk + dmdt*h_in_out - P*vdt - q)  # [K/s]
    
    # Pressure derivative
    dydt[2] = P * ((sum_RYdot/sum_RY) - (vdt/V) + (dydt[0]/T) + (dmdt/M))  # [Pa/s]
    
    # Mass derivative
    dydt[3] = dmdt  # [kg/s]
    
    return dydt

def cycle_ode_intake(t: float, y: np.ndarray, gas: ct.Solution, init: InitialConditions,
                     geom: GeometryParameters, oper: OperatingConditions, 
                     inv: ValveData) -> np.ndarray:
    """
    ODE system for intake phase of engine cycle.
    
    Additional Parameters
    -------------------
    inv : ValveData
        Intake valve data
    """
    # Calculate mass flow through intake valve
    dmdt, flowdir = mair_in(t, oper, gas, inv)
    dmdt *= oper.nvlv  # Account for multiple valves
    
    # Determine properties of incoming/outgoing flow
    if flowdir > 0:
        Y_int = oper.Yin  # Composition of fresh charge
        h_int = oper.Hin  # Enthalpy of fresh charge
    else:
        Y_int = y[4:]  # Current cylinder composition
        h_int = gas.enthalpy_mass  # Current cylinder enthalpy
    
    return _common_ode_calc(t, y, gas, init, geom, oper, dmdt, h_int, Y_int)

def cycle_ode_exhaust(t: float, y: np.ndarray, gas: ct.Solution, init: InitialConditions,
                      geom: GeometryParameters, oper: OperatingConditions, 
                      exv: ValveData) -> np.ndarray:
    """
    ODE system for exhaust phase of engine cycle.
    
    Additional Parameters
    -------------------
    exv : ValveData
        Exhaust valve data
    """
    # Calculate mass flow through exhaust valve
    dmdt, flowdir = mchout(t, oper, gas, exv)
    dmdt = -dmdt * oper.nvlv  # Account for multiple valves, negative for outflow
    
    # Determine properties of incoming/outgoing flow
    if flowdir > 0:
        Y_ex = y[4:]  # Current cylinder composition
        h_ex = gas.enthalpy_mass  # Current cylinder enthalpy
    else:
        Y_ex = oper.Yex  # Composition of exhaust gases
        h_ex = oper.Hex  # Enthalpy of exhaust gases
    
    return _common_ode_calc(t, y, gas, init, geom, oper, dmdt, h_ex, Y_ex)

def vdot1(t: float, init: InitialConditions, geom: GeometryParameters, 
          oper: OperatingConditions) -> float:
    """Calculate rate of volume change."""
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