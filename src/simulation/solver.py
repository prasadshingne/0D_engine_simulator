"""ODE solver for engine simulation."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Callable
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ..engine.geometry import GeometryParams
from ..engine.heat_transfer import HeatTransfer
from ..models.chemistry import Chemistry

@dataclass
class SolverParams:
    """Solver parameters."""
    method: str = "LSODA"    # Integration method (matches original)
    rtol: float = 1.0e-5     # Relative tolerance (balanced)
    atol: float = 1.0e-7     # Absolute tolerance (balanced)
    max_step: float = 5.0e-4  # Maximum step size (increased)
    first_step: float = 1.0e-6  # First step size (increased)
    min_temp: float = 200.0   # Minimum allowed temperature [K]
    max_temp: float = 3500.0  # Maximum allowed temperature [K]
    min_press: float = 1e4    # Minimum allowed pressure [Pa]
    max_press: float = 1e8    # Maximum allowed pressure [Pa]
    min_mass_fraction: float = -1e-10  # Minimum allowed mass fraction before error
    mass_fraction_threshold: float = 1e-12  # Threshold below which to set to zero
    max_rate_limit: float = 1000.0  # Maximum allowed fractional change in mass fraction per step
    
class EngineSolver:
    """Engine cycle ODE solver."""
    
    def __init__(self, geom: GeometryParams, heat_transfer: HeatTransfer,
                 chemistry: Chemistry, params: SolverParams = SolverParams()):
        """
        Initialize solver.
        
        Parameters
        ----------
        geom : GeometryParams
            Engine geometry
        heat_transfer : HeatTransfer
            Heat transfer model
        chemistry : Chemistry
            Chemistry interface
        params : SolverParams, optional
            Solver parameters
        """
        self.geom = geom
        self.heat_transfer = heat_transfer
        self.chemistry = chemistry
        self.params = params
        self.pbar = None
        self.last_t = None  # Track last time for progress updates
        self.t_eval = None  # Store evaluation points
        self.gamma = 1.35   # Cache gamma for motored pressure calc
        
        # Store reference conditions
        self.p_ref = None
        self.T_ref = None
        self.V_ref = None
        
    def _calculate_motored_pressure(self, crank_angle: float) -> float:
        """Calculate motored pressure assuming polytropic compression."""
        V = self.geom.cylinder_volume(crank_angle)
        return self.p_ref * (self.V_ref/V)**self.gamma
    
    def _ode_system(self, t: float, y: np.ndarray, rpm: float, T_wall: float) -> np.ndarray:
        """Calculate derivatives for engine ODE system."""
        # Extract state variables
        T = y[0]              # Temperature [K]
        V = y[1]              # Volume [m³]
        P = y[2]              # Pressure [Pa]
        m = y[3]              # Mass [kg]
        Y = y[4:]            # Species mass fractions [-] (no need to copy)
        
        # Quick bounds check before expensive operations
        if not (200 <= T <= 3500 and 1e4 <= P <= 1e8):
            if T < 200 or T > 3500:
                raise ValueError(f"Temperature {T} K out of bounds [200, 3500]")
            else:
                raise ValueError(f"Pressure {P} Pa out of bounds [1e4, 1e8]")
        
        # Calculate crank angle and volume change
        theta = (rpm * 2 * np.pi / 60) * t  # [rad]
        dVdt = self.geom.volume_rate(theta, rpm)
        
        # Update gas state and get properties
        self.chemistry.update_state(T, P, Y)
        props = self.chemistry.get_properties()
        
        # Get reaction rates and heat release
        mdot, Q_chem = self.chemistry.get_reaction_rates()  # Q_chem in [W/m³]
        ydot = mdot / props['rho']
        
        # Calculate heat transfer only if temperature difference is significant
        if abs(T - T_wall) > 50:  # Skip if ΔT < 50K
            p_motored = self._calculate_motored_pressure(theta)
            Q_wall, _ = self.heat_transfer.heat_transfer_rate(
                theta, rpm, P, T, p_motored,
                self.p_ref, self.T_ref, self.V_ref, T_wall
            )
        else:
            Q_wall = 0.0
        
        # Energy equation
        cv = props['cv']  # cv is already bounded in chemistry
        dTdt = 1.0/(m * cv) * (-P*dVdt + Q_chem*V - Q_wall)
        
        # Pressure equation from ideal gas law
        dPdt = P * (dTdt/T - dVdt/V)
        
        # Combine derivatives (mass is constant)
        dydt = np.zeros_like(y)
        dydt[0] = dTdt
        dydt[1] = dVdt
        dydt[2] = dPdt
        dydt[4:] = ydot
        
        return dydt
    
    def solve_closed_cycle(self, rpm: float, T_wall: float,
                          ca_start: float, ca_end: float,
                          y0: np.ndarray) -> Dict:
        """
        Solve closed cycle from IVC to EVO.
        
        Parameters
        ----------
        rpm : float
            Engine speed [rpm]
        T_wall : float
            Wall temperature [K]
        ca_start : float
            Start crank angle [deg]
        ca_end : float
            End crank angle [deg]
        y0 : np.ndarray
            Initial state vector
            
        Returns
        -------
        Dict
            Solution dictionary with time, states, and crank angles
        """
        # Convert crank angles to radians
        theta_start = np.deg2rad(ca_start)
        theta_end = np.deg2rad(ca_end)
        
        # Calculate time span
        omega = rpm * 2 * np.pi / 60  # [rad/s]
        t_span = (0, (theta_end - theta_start)/omega)
        
        # Store reference conditions at IVC
        self.p_ref = y0[2]
        self.T_ref = y0[0]
        self.V_ref = y0[1]
        
        # Create time evaluation points
        self.t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Reduced from 2000 to 1000 points
        self.last_t = None
        
        # Initialize progress bar
        print("\nSolving engine cycle:")
        self.pbar = tqdm(total=len(self.t_eval), desc="Progress", unit="%")
        
        try:
            # Solve ODE system
            solution = solve_ivp(
                fun=lambda t, y: self._ode_system(t, y, rpm, T_wall),
                t_span=t_span,
                y0=y0,
                method=self.params.method,
                t_eval=self.t_eval,
                rtol=self.params.rtol,
                atol=self.params.atol,
                max_step=self.params.max_step,
                first_step=self.params.first_step
            )
        finally:
            # Clean up
            self.pbar.close()
            self.pbar = None
            self.t_eval = None
            self.last_t = None
        
        # Calculate crank angles
        crank_angles = ca_start + np.rad2deg(omega * solution.t)
        
        return {
            't': solution.t,
            'y': solution.y,
            'ca': crank_angles,
            'success': solution.success,
            'message': solution.message,
            'nfev': solution.nfev,
            'njev': solution.njev
        } 