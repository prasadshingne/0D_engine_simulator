"""Full Cycle (720 CA deg) Simulation for multi-zone HCCI engine."""

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from engine_types import GeometryParameters, OperatingConditions, InitialConditions, ValveData
from cycle_ode_phases import cycle_ode_intake, cycle_ode_exhaust
from cycle_ode import cycle_ode

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
    
    return GeometryParameters(
        a=a,
        conr=conr,
        bore=bore,
        Vcl=Vcl,
        Vd=Vd
    )

def setup_valve_data() -> Tuple[ValveData, ValveData]:
    """Setup intake and exhaust valve data."""
    # Reference diameters
    inv_refd = 31  # Reference diameter for intake valve [mm]
    exv_refd = 25  # Reference diameter for exhaust valve [mm]
    
    # Read valve lift and CD data
    data_dir = Path("input_data")
    
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

def setup_operating_conditions(gas: ct.Solution, inv_dur: float) -> OperatingConditions:
    """
    Setup engine operating conditions.
    
    Parameters
    ----------
    gas : ct.Solution
        Cantera Solution object
    inv_dur : float
        Intake valve duration [deg]
    """
    # Basic parameters
    rpm = 2000  # Engine speed [rev/min]
    omega = 2*np.pi*rpm/60  # Crank Speed [rad/sec]
    
    # Valve timing
    nvo = 100  # Negative Valve Overlap [deg]
    evc = -nvo/2  # Exhaust Valve Closing [deg]
    ivo = nvo/2  # Intake Valve Opening [deg]
    ivc = -(360-nvo/2-inv_dur)+360  # Intake Valve Closing [deg]
    evo = 720 - ivc  # Exhaust Valve Opening [deg]
    
    # Intake conditions - pure air
    pin = 1.01325e5  # Intake Pressure [Pa]
    tin = 300  # Intake Temperature [K]
    compin = {'O2': 0.21, 'N2': 0.79}  # Pure air composition
    
    # Set intake gas state and get properties
    gas.TPX = tin, pin, compin
    Yin = gas.Y
    Hin = gas.h
    
    # Exhaust conditions
    pex = 1.01325e5  # Exhaust Pressure [Pa] - set equal to intake for testing
    tex = 300  # Exhaust Temperature [K] - set equal to intake for testing
    
    # Set exhaust gas state and get properties
    gas.TPX = tex, pex, compin
    Yex = gas.Y
    Hex = gas.h
    
    # Other parameters
    stroke = 0.086  # [m]
    Upbar = 2*stroke*rpm/60  # Mean piston Speed [m/s]
    twall = 300  # Wall temperature [K] - set equal to intake for testing
    
    # Fuel injection parameters - set to 0 for air-only test
    minj = 0.0  # Mass of fuel injected [mg/cycle/cyl]
    injt = 0.0  # Injection timing [deg]
    injdur = 0.0  # Injection duration [deg]
    hvap = 0.0  # Heat of vaporization [J/kg]
    
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
        Hex=Hex,
        minj=minj,
        injt=injt,
        injdur=injdur,
        hvap=hvap
    )

def plot_cycle_results(results: Dict, cycle_num: int, oper: OperatingConditions):
    """Plot results for a single cycle."""
    # Combine all phases
    t_total = []
    T = []
    P = []
    M = []
    V = []
    crank_angle = []
    
    t_offset = 0
    for phase in ['compression1', 'intake', 'compression2', 'exhaust']:
        t = results[phase]['t']
        t_total.extend(t + t_offset)
        T.extend(results[phase]['y'][0])
        P.extend(results[phase]['y'][2])
        M.extend(results[phase]['y'][3])
        V.extend(results[phase]['y'][1])
        crank_angle.extend(oper.soc + oper.rpm/60*360*t)
        t_offset += t[-1]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Temperature plot
    ax1.plot(crank_angle, T)
    ax1.set_xlabel('Crank Angle [deg]')
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('Gas Temperature')
    ax1.grid(True)
    
    # Pressure plot
    ax2.plot(crank_angle, np.array(P)/1e5)  # Convert to bar
    ax2.set_xlabel('Crank Angle [deg]')
    ax2.set_ylabel('Pressure [bar]')
    ax2.set_title('Cylinder Pressure')
    ax2.grid(True)
    
    # Mass plot
    ax3.plot(crank_angle, np.array(M)*1e6)  # Convert to mg
    ax3.set_xlabel('Crank Angle [deg]')
    ax3.set_ylabel('Mass [mg]')
    ax3.set_title('In-Cylinder Mass')
    ax3.grid(True)
    
    # P-V diagram
    ax4.plot(V, np.array(P)/1e5)
    ax4.set_xlabel('Volume [m³]')
    ax4.set_ylabel('Pressure [bar]')
    ax4.set_title('P-V Diagram')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f'Cycle {cycle_num} Results')
    plt.show()

def run_cycle_phase(phase: str, y0: np.ndarray, ca_span: Tuple[float, float],
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
    ca_span : Tuple[float, float]
        Crank angle span for integration [degrees]
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
    # Convert crank angle span to time span based on RPM
    t_span = (0, (ca_span[1] - ca_span[0])/(oper.rpm/60*360))
    
    print(f"\nStarting {phase} phase (CA={ca_span[0]:.1f}° to {ca_span[1]:.1f}°)")
    print(f"Initial conditions: T={y0[0]:.1f}K, P={y0[2]/1e5:.2f}bar, m={y0[3]*1e6:.2f}mg")
    start_time = time.time()
    
    # Create wrapper function that uses crank angle as the primary variable
    def ode_wrapper(t, y):
        # Calculate current crank angle directly
        oper.soc = ca_span[0] + (t/t_span[1]) * (ca_span[1] - ca_span[0])
        
        # Select appropriate ODE function based on phase
        if phase == 'intake':
            return cycle_ode_intake(t, y, gas, init, geom, oper, valve)
        elif phase == 'exhaust':
            return cycle_ode_exhaust(t, y, gas, init, geom, oper, valve)
        else:  # compression/expansion
            return cycle_ode(t, y, gas, init, geom, oper)
    
    # Solve ODEs with appropriate solver settings for each phase
    if phase in ['intake', 'exhaust']:
        # Use Radau for stiff valve flow phases
        sol = solve_ivp(
            ode_wrapper,
            t_span,
            y0,
            method='Radau',
            rtol=1e-6,
            atol=1e-8,
            first_step=1e-8,  # Small first step for valve events
            max_step=1e-4,    # Limit maximum step size
            jac_sparsity=None  # Let solver determine sparsity
        )
    else:
        # Use BDF for compression/expansion phases
        sol = solve_ivp(
            ode_wrapper,
            t_span,
            y0,
            method='BDF',
            rtol=1e-6,
            atol=1e-8
        )
    
    if not sol.success:
        print(f"Warning: {phase} phase solver failed: {sol.message}")
        
    elapsed_time = time.time() - start_time
    print(f"Completed {phase} phase in {elapsed_time:.2f} seconds")
    print(f"Final conditions: T={sol.y[0,-1]:.1f}K, P={sol.y[2,-1]/1e5:.2f}bar, m={sol.y[3,-1]*1e6:.2f}mg")
    print(f"Number of function evaluations: {sol.nfev}")
    print(f"Number of steps: {len(sol.t)}")
    
    # Check for unrealistic values
    if sol.y[0,-1] > 3000 or sol.y[0,-1] < 200:  # Temperature limits
        print(f"Warning: Unrealistic temperature {sol.y[0,-1]:.1f}K")
    if sol.y[2,-1] > 200e5 or sol.y[2,-1] < 0.5e5:  # Pressure limits (0.5-200 bar)
        print(f"Warning: Unrealistic pressure {sol.y[2,-1]/1e5:.2f}bar")
    if sol.y[3,-1] < 0:  # Mass should be positive
        print(f"Warning: Negative mass {sol.y[3,-1]*1e6:.2f}mg")
    
    # Extract end conditions
    return {
        'Tend': sol.y[0, -1],    # Temperature [K]
        'Vend': sol.y[1, -1],    # Volume [m³]
        'Pend': sol.y[2, -1],    # Pressure [Pa]
        'Mend': sol.y[3, -1],    # Mass [kg]
        'Yend': sol.y[4:, -1],   # Species mass fractions [-]
        't': sol.t,              # Time points [s]
        'y': sol.y,              # All solution data
        'ca': np.linspace(ca_span[0], ca_span[1], len(sol.t))  # Crank angle points [deg]
    }

def main():
    """Main function to run the engine simulation."""
    # Setup geometry and valve data
    geom = setup_geometry()
    inv, exv = setup_valve_data()
    
    # Initialize gas with mechanism - using gri30 for air simulation
    gas = ct.Solution('gri30.yaml')
    
    # Setup operating conditions
    oper = setup_operating_conditions(gas, inv.dur)
    
    # Initial conditions at start of cycle (EVC)
    init = InitialConditions(theta=oper.soc * np.pi/180)
    
    # Set initial gas state
    gas.TPX = oper.tin, oper.pin, oper.compin
    
    # Calculate initial volume and mass
    x0 = (geom.a * np.cos(init.theta) + 
          np.sqrt(geom.conr**2 - (geom.a * np.sin(init.theta))**2))
    V0 = geom.Vcl + np.pi/4 * geom.bore**2 * (geom.conr + geom.a - x0)
    M0 = V0 * gas.density
    
    print("\nEngine Geometry:")
    print(f"Bore = {geom.bore*1000:.1f} mm")
    print(f"Stroke = {geom.a*2*1000:.1f} mm")
    print(f"Connecting rod = {geom.conr*1000:.1f} mm")
    print(f"Compression ratio = {(geom.Vcl + geom.Vd)/geom.Vcl:.1f}")
    print(f"Displacement volume = {geom.Vd*1e6:.1f} cm³")
    print(f"Clearance volume = {geom.Vcl*1e6:.1f} cm³")
    
    print("\nValve Timing:")
    print(f"EVC = {oper.evc:.1f}°")
    print(f"IVO = {oper.ivo:.1f}°")
    print(f"IVC = {oper.ivc:.1f}°")
    print(f"EVO = {oper.evo:.1f}°")
    
    print(f"\nInitial conditions:")
    print(f"V0 = {V0*1e6:.2f} cm³")
    print(f"M0 = {M0*1e6:.2f} mg")
    print(f"Starting crank angle = {oper.soc:.1f}°")
    print(f"Gas composition: {oper.compin}")
    
    # Initial state vector
    y0 = np.zeros(4 + gas.n_species)
    y0[0] = oper.tin      # Temperature [K]
    y0[1] = V0           # Volume [m³]
    y0[2] = oper.pin     # Pressure [Pa]
    y0[3] = M0           # Mass [kg]
    y0[4:] = gas.Y       # Species mass fractions [-]
    
    print("\nStarting single cycle air-only simulation")
    print("Cycle consists of: EVC→IVO (compression) → IVO→IVC (intake) → IVC→EVO (compression) → EVO→EVC (exhaust)")
    
    cycle_results = {}
    
    # EVC to IVO (compression)
    ca_span = (oper.evc, oper.ivo)
    res1 = run_cycle_phase('compression1', y0, ca_span, gas, init, geom, oper)
    cycle_results['compression1'] = res1
    
    # IVO to IVC (intake)
    oper.soc = oper.ivo
    init.theta = oper.soc * np.pi/180
    ca_span = (oper.ivo, oper.ivc)
    res2 = run_cycle_phase('intake', res1['y'][:,-1], ca_span, gas, init, geom, oper, inv)
    cycle_results['intake'] = res2
    
    # IVC to EVO (compression)
    oper.soc = oper.ivc
    init.theta = oper.soc * np.pi/180
    ca_span = (oper.ivc, oper.evo)
    res3 = run_cycle_phase('compression2', res2['y'][:,-1], ca_span, gas, init, geom, oper)
    cycle_results['compression2'] = res3
    
    # EVO to EVC (exhaust)
    oper.soc = oper.evo
    init.theta = oper.soc * np.pi/180
    ca_span = (oper.evo, oper.evc + 360)
    res4 = run_cycle_phase('exhaust', res3['y'][:,-1], ca_span, gas, init, geom, oper, exv)
    cycle_results['exhaust'] = res4
    
    # Plot results
    plot_cycle_results(cycle_results, 1, oper)
    
    return cycle_results

if __name__ == "__main__":
    results = main() 