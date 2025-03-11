import numpy as np
from typing import Tuple, Dict
from engine_types import OperatingConditions, ValveData, Gas

def _calculate_valve_flow(t: float, oper: OperatingConditions, gas: Gas, valve: ValveData,
                         upstream_p: float, upstream_t: float, upstream_comp: Dict[str, float],
                         downstream_p: float) -> Tuple[float, int]:
    """
    Common valve flow calculation logic for both inlet and exhaust valves.
    
    Args:
        t: Time [s]
        oper: Operating conditions
        gas: Gas object containing thermodynamic properties
        valve: Valve data
        upstream_p: Upstream pressure [Pa]
        upstream_t: Upstream temperature [K]
        upstream_comp: Upstream composition
        downstream_p: Downstream pressure [Pa]
    
    Returns:
        Tuple of (mass flow rate [kg/s], flow direction [-1 or 1])
    """
    # Calculate crank angle
    CAD = oper.soc + oper.rpm/60*360*t
    
    # Find instantaneous lift and CD
    L = np.interp(CAD, valve.ca, valve.lift)
    LD = L/valve.refd
    CD = np.interp(LD, valve.ra, valve.cd)
    
    # Reference area
    AR = np.pi/4*(valve.refd/1e3)**2
    
    # Set upstream conditions
    gas.set(T=upstream_t, P=upstream_p, composition=upstream_comp)
    upstream_props = {
        'density': gas.density(),
        'temperature': upstream_t,
        'cp': gas.cp_mass(),
        'cv': gas.cv_mass()
    }
    
    # Determine flow direction and properties
    if upstream_p >= downstream_p:
        Pr = downstream_p/upstream_p
        rho0 = upstream_props['density']
        T0 = upstream_props['temperature']
        gamma = upstream_props['cp']/upstream_props['cv']
        R = upstream_props['cp'] - upstream_props['cv']
        flowdir = 1
    else:
        Pr = upstream_p/downstream_p
        # Use downstream properties
        rho0 = gas.density()
        T0 = gas.temperature()
        gamma = gas.cp_mass()/gas.cv_mass()
        R = gas.cp_mass() - gas.cv_mass()
        flowdir = -1
    
    # Initialize mass flow rate
    mass_flow = 0.0
    
    if L > 0:
        # Flow through an orifice relationships
        if Pr <= (2/(gamma+1))**(gamma/(gamma-1)):
            # Choked flow
            rhois = rho0*(2/(gamma+1))**(1/(gamma-1))
            Uis = np.sqrt(gamma*R*T0)*(2/(gamma+1))**0.5
        else:
            rhois = rho0*(Pr)**(1/gamma)
            Uis = np.sqrt(R*T0)*(2*gamma/(gamma-1)*(1-Pr**((gamma-1)/gamma)))**0.5
        
        mass_flow = flowdir*CD*AR*rhois*Uis
    
    return mass_flow, flowdir

def mair_in(t: float, oper: OperatingConditions, gas: Gas, inv: ValveData) -> Tuple[float, int]:
    """Calculate mass flow rate through inlet valve"""
    return _calculate_valve_flow(
        t=t,
        oper=oper,
        gas=gas,
        valve=inv,
        upstream_p=oper.pin,
        upstream_t=oper.tin,
        upstream_comp=oper.compin,
        downstream_p=gas.pressure()
    )

def mchout(t: float, oper: OperatingConditions, gas: Gas, exv: ValveData) -> Tuple[float, int]:
    """Calculate mass flow rate through exhaust valve"""
    return _calculate_valve_flow(
        t=t,
        oper=oper,
        gas=gas,
        valve=exv,
        upstream_p=gas.pressure(),
        upstream_t=gas.temperature(),
        upstream_comp=oper.compin,
        downstream_p=oper.pex
    ) 