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
    # Debug output - only every 10 degrees
    debug = (abs(round(crank_angle) % 10) == 0)
    
    # Normalize crank angle to 0-720 range
    crank_angle = crank_angle % 720
    
    if debug:
        print(f"\nValve flow calculation at CA={crank_angle:.1f}°:")
        print(f"Upstream: P={upstream_p/1e5:.2f}bar, T={upstream_T:.1f}K")
        print(f"Downstream: P={downstream_p/1e5:.2f}bar")
    
    # Ensure pressures and temperatures are reasonable
    if upstream_p > 200e5:  # 200 bar max
        print(f"Warning: Limiting upstream pressure from {upstream_p/1e5:.1f} to 200 bar")
        upstream_p = 200e5
    if downstream_p > 200e5:
        print(f"Warning: Limiting downstream pressure from {downstream_p/1e5:.1f} to 200 bar")
        downstream_p = 200e5
    if upstream_T > 3000:  # 3000K max
        print(f"Warning: Limiting upstream temperature from {upstream_T:.1f} to 3000K")
        upstream_T = 3000
        
    # Ensure pressures are positive and not too small
    eps = 0.5e5  # 0.5 bar minimum
    upstream_p = max(upstream_p, eps)
    downstream_p = max(downstream_p, eps)
    upstream_T = max(upstream_T, 200)  # 200K minimum
    
    # Get valve lift at current crank angle
    lift = np.interp(crank_angle, valve.ca, valve.lift)  # [mm]
    
    # If valve is effectively closed, return zero flow
    if lift < 0.01:  # Less than 0.01mm lift
        if debug:
            print(f"Valve effectively closed (lift = {lift:.3f}mm)")
        return 0.0, 1
    
    # Calculate lift/diameter ratio and get discharge coefficient
    ld_ratio = lift / valve.refd  # [-]
    cd = np.interp(ld_ratio, valve.ra, valve.cd)  # [-]
    
    # Limit cd to reasonable values
    cd = max(min(cd, 1.0), 0.0)
    
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
    pr = max(pr, 0.1)  # Limit minimum pressure ratio to 0.1
    
    # Critical pressure ratio (assuming gamma = 1.4)
    pr_crit = 0.528  # (2/(1.4+1))**(1.4/(1.4-1))
    
    try:
        # Get gas constant for air (J/kg·K)
        R = ct.gas_constant / gas.mean_molecular_weight
        
        # Calculate mass flow rate
        if pr > pr_crit:  # Subsonic flow
            mf = cd * area * upstream_p * np.sqrt(2*1.4/(1.4-1)/R/upstream_T * 
                 (pr**(2/1.4) - pr**((1.4+1)/1.4)))
        else:  # Choked flow
            mf = cd * area * upstream_p * np.sqrt(1.4 * (2/(1.4+1))**((1.4+1)/(1.4-1)) / 
                 (R * upstream_T))
        
        # Limit mass flow to reasonable values
        max_flow = 0.1  # kg/s - reasonable maximum for this engine size
        mf = max(min(mf, max_flow), -max_flow)
        
        if debug:
            print(f"Lift={lift:.3f}mm, CD={cd:.3f}")
            print(f"Area={area*1e6:.2f}mm², PR={pr:.3f}")
            print(f"Mass flow={mf*1000:.2f}g/s, Direction={flowdir}")
        
    except (ValueError, RuntimeWarning) as e:
        print(f"Warning in valve flow calculation: {e}")
        print(f"Values: p_up={upstream_p}, p_down={downstream_p}, T={upstream_T}, pr={pr}")
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