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

def setup_operating_conditions(gas: ct.Solution) -> OperatingConditions:
    """Setup engine operating conditions."""
    # Engine speed
    rpm = 2000  # [rev/min]
    
    # Valve timing for closed cycle
    ivc = -180  # [deg] before TDC
    evo = 180   # [deg] after TDC
    
    # Initial conditions at IVC
    pin = 1.0e5  # [Pa]
    tin = 400  # [K]
    
    # Set up fuel composition (pure iso-octane)
    # First, set the mixture to stoichiometric
    gas.set_equivalence_ratio(1.0, 'C8H18:1', 'O2:1, N2:3.76')
    
    # Now set to phi = 0.8
    gas.set_equivalence_ratio(0.7, 'C8H18:1', 'O2:1, N2:3.76')
    Yin_fresh = gas.Y
    
    # Get mole fractions for the fresh charge
    compin = {}
    for species in ['C8H18', 'O2', 'N2']:  # Removed C7H16
        compin[species] = float(gas[species].X)  # Convert to float to ensure it's JSON serializable
    
    # Set up residual gas composition (burned gas at phi = 0.8)
    gas.equilibrate('HP')
    Yin_residual = gas.Y
    
    # Mix fresh charge and residual (RGF = 30%)
    Yin = 0.70 * Yin_fresh + 0.30 * Yin_residual
    
    # Calculate mixture enthalpy
    gas.TPY = tin, pin, Yin
    Hin = gas.enthalpy_mass
    
    # Wall temperature
    twall = 400  # [K]
    
    # Mean piston speed
    stroke = 0.086  # [m]
    Upbar = 2 * stroke * rpm / 60  # [m/s]
    
    return OperatingConditions(
        rpm=rpm,
        pin=pin,
        tin=tin,
        pex=pin,  # Not used in closed cycle
        tex=tin,  # Not used in closed cycle
        ivo=0,    # Not used in closed cycle
        ivc=ivc,
        evo=evo,
        evc=0,    # Not used in closed cycle
        nvlv=2,   # Not used in closed cycle
        Yin=Yin,
        Hin=Hin,
        twall=twall,
        Upbar=Upbar,
        soc=ivc,  # Start at IVC
        compin=compin  # Fresh charge composition
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
    
    # Use LSODA solver which automatically switches between stiff/non-stiff methods
    sol = solve_ivp(
        ode_wrapper,
        t_span,
        y0,
        method='LSODA',       # Automatically switches between stiff/non-stiff methods
        rtol=1e-4,           # Moderate relative tolerance
        atol=1e-6,           # Moderate absolute tolerance
        max_step=1e-3,       # Reasonable maximum step size
        first_step=1e-6,     # Reasonable first step size
        dense_output=True,   # Enable dense output for smoother curves
        t_eval=np.linspace(t_span[0], t_span[1], 200)  # Keep same number of output points
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
    """Main function to run the closed cycle simulation (IVC to EVO)."""
    # Setup geometry
    geom = setup_geometry()
    
    # Initialize gas with Nissan mechanism
    gas = ct.Solution('input_data/Nissan_chem.yaml')
    
    # Setup operating conditions
    oper = setup_operating_conditions(gas)
    
    # Initial conditions at IVC
    init = InitialConditions(theta=oper.ivc * np.pi/180)  # Convert to radians
    
    # Set initial gas state
    gas.TPY = oper.tin, oper.pin, oper.Yin
    
    # Calculate initial volume at IVC
    x0 = (geom.a * np.cos(init.theta) + 
          np.sqrt(geom.conr**2 - (geom.a * np.sin(init.theta))**2))
    V0 = geom.Vcl + np.pi/4 * geom.bore**2 * (geom.conr + geom.a - x0)
    M0 = V0 * gas.density
    
    print("\nEngine Geometry:")
    print(f"Bore = {geom.bore*1000:.1f} mm")
    print(f"Stroke = {geom.a*2*1000:.1f} mm")
    print(f"Connecting rod = {geom.conr*1000:.1f} mm")
    print(f"Compression ratio = {(geom.Vcl + geom.Vd)/geom.Vcl:.1f}")
    
    print("\nSimulating closed cycle (IVC to EVO):")
    print(f"IVC = {oper.ivc:.1f}°")
    print(f"EVO = {oper.evo:.1f}°")
    print(f"Initial conditions at IVC:")
    print(f"Temperature = {oper.tin:.1f} K")
    print(f"Pressure = {oper.pin/1e5:.2f} bar")
    print(f"Volume = {V0*1e6:.2f} cm³")
    print(f"Mass = {M0*1e6:.2f} mg")
    
    # Initial state vector
    y0 = np.zeros(gas.n_species + 4)
    y0[0] = oper.tin  # Temperature [K]
    y0[1] = V0        # Volume [m³]
    y0[2] = oper.pin  # Pressure [Pa]
    y0[3] = M0        # Mass [kg]
    y0[4:] = oper.Yin # Species mass fractions [-]
    
    # Run closed cycle simulation
    results = run_cycle_phase(
        'closed_cycle',
        y0,
        (oper.ivc, oper.evo),
        gas,
        init,
        geom,
        oper
    )
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Get crank angles for x-axis
    crank_angle = np.linspace(oper.ivc, oper.evo, len(results['t']))
    
    # Temperature plot
    ax1.plot(crank_angle, results['y'][0])
    ax1.set_xlabel('Crank Angle [deg]')
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('Gas Temperature')
    ax1.grid(True)
    
    # Pressure plot
    ax2.plot(crank_angle, np.array(results['y'][2])/1e5)  # Convert to bar
    ax2.set_xlabel('Crank Angle [deg]')
    ax2.set_ylabel('Pressure [bar]')
    ax2.set_title('Cylinder Pressure')
    ax2.grid(True)
    
    # Mass plot
    ax3.plot(crank_angle, np.array(results['y'][3])*1e6)  # Convert to mg
    ax3.set_xlabel('Crank Angle [deg]')
    ax3.set_ylabel('Mass [mg]')
    ax3.set_title('In-Cylinder Mass')
    ax3.grid(True)
    
    # P-V diagram
    ax4.plot(results['y'][1], np.array(results['y'][2])/1e5)
    ax4.set_xlabel('Volume [m³]')
    ax4.set_ylabel('Pressure [bar]')
    ax4.set_title('P-V Diagram')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Closed Cycle (IVC to EVO)')
    
    # Save the figure
    plt.savefig('main_combustion_cycle.png', dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    return results

if __name__ == "__main__":
    results = main() 