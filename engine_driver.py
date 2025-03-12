"""Full Cycle (720 CA deg) Simulation for multi-zone HCCI engine."""

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from engine_types import GeometryParameters, OperatingConditions, InitialConditions, ValveData
from cycle_ode_phases import cycle_ode_intake, cycle_ode_exhaust, cycle_ode_new

def setup_geometry() -> GeometryParameters:
    """Setup engine geometry parameters."""
    # Bore X Stroke
    bore = 0.086  # bore diameter [m]
    stroke = 0.086  # stroke length [m]
    a = stroke/2  # crank length [m]
    
    # Geometric Compression Ratio
    cr = 12.5  # Geometric Compression Ratio
    tdcl = stroke/(cr-1)  # TDC Clearance Length [m]
    
    # Con-Rod Length
    conr = 0.1455  # Connecting rod length [m]
    
    # Derived parameters
    Vd = np.pi/4 * bore**2 * stroke  # Displaced volume [m³]
    Vcl = np.pi/4 * bore**2 * tdcl  # Clearance volume [m³]
    Vt = Vd + Vcl  # Total cylinder volume [m³]
    
    return GeometryParameters(
        a=a,
        conr=conr,
        bore=bore
    )

def setup_valve_data() -> Tuple[ValveData, ValveData]:
    """Setup intake and exhaust valve data."""
    # Reference diameters
    inv_refd = 31  # Reference diameter for intake valve [mm]
    exv_refd = 25  # Reference diameter for exhaust valve [mm]
    
    # Read valve lift and CD data
    data_dir = Path("valve_data")
    
    # Intake valve data
    inv_ca, inv_lift = np.loadtxt(data_dir / "IL.txt", delimiter='\t', unpack=True)
    inv_ra, inv_cd = np.loadtxt(data_dir / "CDIL.txt", delimiter='\t', unpack=True)
    
    # Exhaust valve data
    exv_ca, exv_lift = np.loadtxt(data_dir / "EL.txt", delimiter='\t', unpack=True)
    exv_ra, exv_cd = np.loadtxt(data_dir / "CDEL.txt", delimiter='\t', unpack=True)
    
    # Calculate durations
    inv_dur = exv_ca[-1] - exv_ca[0]
    exv_dur = inv_dur
    
    return (
        ValveData(ca=inv_ca, lift=inv_lift, refd=inv_refd, ra=inv_ra, cd=inv_cd, dur=inv_dur),
        ValveData(ca=exv_ca, lift=exv_lift, refd=exv_refd, ra=exv_ra, cd=exv_cd, dur=exv_dur)
    )

def setup_operating_conditions(gas: ct.Solution) -> OperatingConditions:
    """Setup engine operating conditions."""
    # Basic parameters
    rpm = 2000  # Engine speed [rev/min]
    omega = 2*np.pi*rpm/60  # Crank Speed [rad/sec]
    
    # Valve timing
    nvo = 100  # Negative Valve Overlap [deg]
    evc = -nvo/2  # Exhaust Valve Closing [deg]
    ivo = nvo/2  # Intake Valve Opening [deg]
    ivc = -(360-nvo/2-inv_dur)+360  # Intake Valve Closing [deg]
    evo = 720 - ivc  # Exhaust Valve Opening [deg]
    
    # Intake conditions
    pin = 1.01325e5  # Intake Pressure [Pa]
    tin = 450  # Intake Temperature [K]
    compin = {'C8H18': 0, 'O2': 1, 'N2': 3.76, 'CO2': 0, 'H2O': 0}
    
    # Set intake gas state and get properties
    gas.TPX = tin, pin, compin
    Yin = gas.Y
    Hin = gas.h
    
    # Exhaust conditions
    pex = 1.08 * 1.01325e5  # Exhaust Pressure [Pa]
    tex = 550  # Exhaust Temperature [K]
    
    # Set exhaust gas state and get properties
    gas.TPX = tex, pex, compin
    Yex = gas.Y
    Hex = gas.h
    
    # Other parameters
    Upbar = 2*stroke*rpm/60  # Mean piston Speed [m/s]
    twall = 353  # Wall temperature [K]
    
    return OperatingConditions(
        soc=evc,  # Start at EVC
        rpm=rpm,
        tin=tin,
        pin=pin,
        tex=tex,
        pex=pex,
        compin=compin,
        twall=twall,
        Upbar=Upbar,
        evc=evc,
        ivo=ivo,
        ivc=ivc,
        evo=evo,
        Yin=Yin,
        Hin=Hin,
        Yex=Yex,
        Hex=Hex
    )

def run_cycle_phase(phase: str, y0: np.ndarray, t_span: Tuple[float, float],
                    gas: ct.Solution, init: InitialConditions, geom: GeometryParameters,
                    oper: OperatingConditions, valve: Optional[ValveData] = None) -> Dict:
    """
    Run a single phase of the engine cycle.
    
    Parameters
    ----------
    phase : str
        Phase name: 'compression', 'intake', or 'exhaust'
    y0 : np.ndarray
        Initial conditions
    t_span : Tuple[float, float]
        Time span for integration
    gas : ct.Solution
        Cantera Solution object
    init : InitialConditions
        Initial conditions
    geom : GeometryParameters
        Geometry parameters
    oper : OperatingConditions
        Operating conditions
    valve : Optional[ValveData]
        Valve data for intake/exhaust phases
        
    Returns
    -------
    Dict
        Results dictionary containing end conditions
    """
    # Select appropriate ODE function based on phase
    if phase == 'intake':
        ode_fn = lambda t, y: cycle_ode_intake(t, y, gas, init, geom, oper, valve)
    elif phase == 'exhaust':
        ode_fn = lambda t, y: cycle_ode_exhaust(t, y, gas, init, geom, oper, valve)
    else:  # compression/expansion
        ode_fn = lambda t, y: cycle_ode_new(t, y, gas, init, geom, oper)
    
    # Solve ODEs
    sol = solve_ivp(
        ode_fn,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-5,
        atol=1e-12
    )
    
    # Extract end conditions
    return {
        'Tend': sol.y[0, -1],    # Temperature [K]
        'Vend': sol.y[1, -1],    # Volume [m³]
        'Pend': sol.y[2, -1],    # Pressure [Pa]
        'Mend': sol.y[3, -1],    # Mass [kg]
        'Yend': sol.y[4:, -1],   # Species mass fractions [-]
        't': sol.t,              # Time points [s]
        'y': sol.y               # All solution data
    }

def main():
    """Main function to run the engine simulation."""
    # Setup geometry and valve data
    geom = setup_geometry()
    inv, exv = setup_valve_data()
    
    # Initialize gas with mechanism
    gas = ct.Solution('nissan_chem.xml')
    
    # Setup operating conditions
    oper = setup_operating_conditions(gas)
    
    # Initial conditions at start of cycle (EVC)
    init = InitialConditions(theta=oper.soc * np.pi/180)
    
    # Set initial gas state
    gas.TPX = oper.tin, oper.pin, oper.compin
    
    # Calculate initial volume and mass
    x0 = (geom.a * np.cos(init.theta) + 
          np.sqrt(geom.conr**2 - (geom.a * np.sin(init.theta))**2))
    V0 = geom.Vcl + np.pi/4 * geom.bore**2 * (geom.conr + geom.a - x0)
    M0 = V0 * gas.density
    
    # Initial state vector
    y0 = np.zeros(4 + gas.n_species)
    y0[0] = oper.tin      # Temperature [K]
    y0[1] = V0           # Volume [m³]
    y0[2] = oper.pin     # Pressure [Pa]
    y0[3] = M0           # Mass [kg]
    y0[4:] = gas.Y       # Species mass fractions [-]
    
    # Run multiple cycles
    n_cycles = 5
    results = []
    
    for cycle in range(n_cycles):
        print(f"Running cycle {cycle+1}")
        
        # EVC to IVO (compression)
        t_span = (0, (oper.ivo - oper.evc)/(oper.rpm/60*360))
        res1 = run_cycle_phase('compression', y0, t_span, gas, init, geom, oper)
        
        # IVO to IVC (intake)
        oper.soc = oper.ivo
        init.theta = oper.soc * np.pi/180
        t_span = (0, (oper.ivc - oper.ivo)/(oper.rpm/60*360))
        res2 = run_cycle_phase('intake', res1['y'][:,-1], t_span, gas, init, geom, oper, inv)
        
        # IVC to EVO (compression + combustion)
        oper.soc = oper.ivc
        init.theta = oper.soc * np.pi/180
        t_span = (0, (oper.evo - oper.ivc)/(oper.rpm/60*360))
        res3 = run_cycle_phase('compression', res2['y'][:,-1], t_span, gas, init, geom, oper)
        
        # EVO to EVC (exhaust)
        oper.soc = oper.evo
        init.theta = oper.soc * np.pi/180
        t_span = (0, (oper.evc + 360 - oper.evo)/(oper.rpm/60*360))
        res4 = run_cycle_phase('exhaust', res3['y'][:,-1], t_span, gas, init, geom, oper, exv)
        
        # Store results
        results.append({
            'compression1': res1,
            'intake': res2,
            'compression2': res3,
            'exhaust': res4
        })
        
        # Update initial conditions for next cycle
        y0 = res4['y'][:,-1]
    
    return results

if __name__ == "__main__":
    results = main() 