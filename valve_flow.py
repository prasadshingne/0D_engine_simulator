import numpy as np
from typing import Tuple, Dict
import cantera as ct
from engine_types import OperatingConditions, ValveData, Gas

def _calculate_valve_flow(upstream_p: float, downstream_p: float, upstream_T: float,
                      gas: ct.Solution, valve: ValveData, crank_angle: float) -> Tuple[float, int]:
    """
    Calculate mass flow rate through a valve.
    
    Parameters
    ----------
    upstream_p : float
        Upstream pressure [Pa]
    downstream_p : float
        Downstream pressure [Pa]
    upstream_T : float
        Upstream temperature [K]
    gas : ct.Solution
        Gas object
    valve : ValveData
        Valve data
    crank_angle : float
        Current crank angle [deg]
    
    Returns
    -------
    Tuple[float, int]
        Mass flow rate [kg/s], Flow direction (1: forward, -1: reverse)
    """
    # Debug output only at major crank angles
    debug = (abs(round(crank_angle)) % 90 == 0)
    
    # Normalize crank angle to 0-720 range
    crank_angle = crank_angle % 720
    
    # Silently limit pressures and temperatures to reasonable values
    upstream_p = min(max(upstream_p, 0.5e5), 200e5)   # 0.5-200 bar
    downstream_p = min(max(downstream_p, 0.5e5), 200e5)
    upstream_T = min(max(upstream_T, 200), 3000)      # 200-3000 K
    
    # Get valve lift at current crank angle
    lift = np.interp(crank_angle, valve.ca, valve.lift)  # [mm]
    
    # If valve is effectively closed, return zero flow
    if lift < 0.01:  # Less than 0.01mm lift
        return 0.0, 1
    
    # Calculate lift/diameter ratio and get discharge coefficient
    ld_ratio = lift / valve.refd  # [-]
    cd = min(max(np.interp(ld_ratio, valve.ra, valve.cd), 0.0), 1.0)  # [-]
    
    # Calculate reference area
    area = np.pi * (valve.refd/1000)**2 / 4  # [m²]
    
    # Calculate pressure ratio and flow direction
    if upstream_p >= downstream_p:
        pr = downstream_p / upstream_p
        flowdir = 1
    else:
        pr = upstream_p / downstream_p
        flowdir = -1
    
    # Ensure pressure ratio is not too small
    pr = max(pr, 0.1)
    
    # Critical pressure ratio (assuming gamma = 1.4)
    pr_crit = 0.528
    
    try:
        # Get gas constant
        R = ct.gas_constant / gas.mean_molecular_weight
        
        # Calculate mass flow rate
        if pr > pr_crit:  # Subsonic flow
            mf = cd * area * upstream_p * np.sqrt(2*1.4/(1.4-1)/R/upstream_T * 
                 (pr**(2/1.4) - pr**((1.4+1)/1.4)))
        else:  # Choked flow
            mf = cd * area * upstream_p * np.sqrt(1.4 * (2/(1.4+1))**((1.4+1)/(1.4-1)) / 
                 (R * upstream_T))
        
        # Limit mass flow to reasonable values
        mf = min(max(mf, -0.1), 0.1)  # ±0.1 kg/s limit
        
        if debug:
            print(f"\nValve at {crank_angle:.0f}°: lift={lift:.2f}mm, flow={mf*1000:.1f}g/s")
        
    except Exception as e:
        print(f"Error in valve flow calculation at {crank_angle:.0f}°: {str(e)}")
        mf = 0.0
    
    return flowdir * mf, flowdir

def mair_in(t: float, oper: OperatingConditions, gas: ct.Solution, 
            valve: ValveData) -> Tuple[float, int]:
    """
    Calculate intake mass flow rate.
    
    Parameters
    ----------
    t : float
        Time [s]
    oper : OperatingConditions
        Operating conditions
    gas : ct.Solution
        Gas object
    valve : ValveData
        Valve data
    
    Returns
    -------
    Tuple[float, int]
        Mass flow rate [kg/s], Flow direction
    """
    # Use current crank angle from operating conditions
    theta = oper.soc  # [deg]
    
    # Calculate mass flow rate
    return _calculate_valve_flow(oper.pin, gas.P, oper.tin, gas, valve, theta)

def mchout(t: float, oper: OperatingConditions, gas: ct.Solution, 
           valve: ValveData) -> Tuple[float, int]:
    """
    Calculate exhaust mass flow rate.
    
    Parameters
    ----------
    t : float
        Time [s]
    oper : OperatingConditions
        Operating conditions
    gas : ct.Solution
        Gas object
    valve : ValveData
        Valve data
    
    Returns
    -------
    Tuple[float, int]
        Mass flow rate [kg/s], Flow direction
    """
    # Use current crank angle from operating conditions
    theta = oper.soc  # [deg]
    
    # Calculate mass flow rate
    return _calculate_valve_flow(gas.P, oper.pex, gas.T, gas, valve, theta) 