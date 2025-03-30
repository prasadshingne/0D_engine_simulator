"""Heat transfer calculations using Woschni correlation."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from .geometry import GeometryParams

@dataclass
class WoschniParams:
    """Woschni correlation parameters."""
    C1: float = 2.28        # Mean piston speed coefficient
    C2: float = 0.00324     # Combustion term coefficient
    C: float = 3.26         # Overall coefficient
    
    # Exponents
    a_bore: float = -0.2    # Bore diameter
    a_press: float = 0.8    # Pressure
    a_temp: float = -0.53   # Temperature
    a_vel: float = 0.8      # Velocity

class HeatTransfer:
    """Heat transfer model using Woschni correlation."""
    
    def __init__(self, geom: GeometryParams, params: WoschniParams = WoschniParams()):
        """
        Initialize heat transfer model.
        
        Parameters
        ----------
        geom : GeometryParams
            Engine geometry parameters
        params : WoschniParams, optional
            Woschni correlation parameters
        """
        self.geom = geom
        self.params = params
        
    def _characteristic_velocity(self, crank_angle: float, rpm: float,
                               pressure: float, temperature: float,
                               p_motored: float, p_ref: float,
                               T_ref: float, V_ref: float) -> float:
        """
        Calculate characteristic velocity for Woschni correlation.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
        rpm : float
            Engine speed [rev/min]
        pressure : float
            Cylinder pressure [Pa]
        temperature : float
            Gas temperature [K]
        p_motored : float
            Motored pressure at same crank angle [Pa]
        p_ref : float
            Reference pressure at IVC [Pa]
        T_ref : float
            Reference temperature at IVC [K]
        V_ref : float
            Reference volume at IVC [m³]
            
        Returns
        -------
        float
            Characteristic velocity [m/s]
        """
        # Mean piston speed
        Up = 2 * self.geom.stroke * rpm / 60  # [m/s]
        
        # Base term using mean piston speed
        w = self.params.C1 * Up
        
        # Add combustion term after TDC
        if crank_angle > 0:
            w += (self.params.C2 * self.geom.displacement * T_ref /
                 (p_ref * V_ref) * (pressure - p_motored))
            
        return w
    
    def heat_transfer_coefficient(self, crank_angle: float, rpm: float,
                                pressure: float, temperature: float,
                                p_motored: float, p_ref: float,
                                T_ref: float, V_ref: float) -> float:
        """
        Calculate heat transfer coefficient using Woschni correlation.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
        rpm : float
            Engine speed [rev/min]
        pressure : float
            Cylinder pressure [Pa]
        temperature : float
            Gas temperature [K]
        p_motored : float
            Motored pressure at same crank angle [Pa]
        p_ref : float
            Reference pressure at IVC [Pa]
        T_ref : float
            Reference temperature at IVC [K]
        V_ref : float
            Reference volume at IVC [m³]
            
        Returns
        -------
        float
            Heat transfer coefficient [W/(m²·K)]
        """
        # Ensure positive values for physical quantities
        pressure = max(pressure, 1e4)  # Minimum 0.1 bar
        temperature = max(temperature, 200.0)  # Minimum 200 K
        p_motored = max(p_motored, 1e4)
        p_ref = max(p_ref, 1e4)
        T_ref = max(T_ref, 200.0)
        
        # Calculate characteristic velocity
        w = self._characteristic_velocity(
            crank_angle, rpm, pressure, temperature,
            p_motored, p_ref, T_ref, V_ref
        )
        w = max(w, 1.0)  # Ensure positive velocity
        
        # Calculate heat transfer coefficient
        # h = C·d^a_bore·p^a_press·T^a_temp·w^a_vel
        h = (self.params.C * 
             self.geom.bore**self.params.a_bore *
             (pressure/1e5)**self.params.a_press *
             temperature**self.params.a_temp *
             w**self.params.a_vel)
        
        # Ensure reasonable coefficient value
        h = max(h, 1.0)  # Minimum 1 W/(m²·K)
        h = min(h, 1e4)  # Maximum 10 kW/(m²·K)
        
        return h
    
    def heat_transfer_rate(self, crank_angle: float, rpm: float,
                          pressure: float, temperature: float,
                          p_motored: float, p_ref: float,
                          T_ref: float, V_ref: float,
                          T_wall: float) -> Tuple[float, float]:
        """
        Calculate heat transfer rate.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
        rpm : float
            Engine speed [rev/min]
        pressure : float
            Cylinder pressure [Pa]
        temperature : float
            Gas temperature [K]
        p_motored : float
            Motored pressure at same crank angle [Pa]
        p_ref : float
            Reference pressure at IVC [Pa]
        T_ref : float
            Reference temperature at IVC [K]
        V_ref : float
            Reference volume at IVC [m³]
        T_wall : float
            Wall temperature [K]
            
        Returns
        -------
        Tuple[float, float]
            Heat transfer rate [W], Heat transfer coefficient [W/(m²·K)]
        """
        # Calculate heat transfer coefficient with safeguards
        h = self.heat_transfer_coefficient(
            crank_angle, rpm, pressure, temperature,
            p_motored, p_ref, T_ref, V_ref
        )
        
        # Calculate surface areas
        head_area, piston_area, liner_area = self.geom.surface_area(crank_angle)
        total_area = max(head_area + piston_area + liner_area, 1e-6)  # Prevent zero area
        
        # Calculate heat transfer rate (positive = loss to walls)
        # Limit temperature difference to prevent excessive heat transfer
        dT = min(max(temperature - T_wall, -2000.0), 2000.0)
        Q = h * total_area * dT
        
        # Limit maximum heat transfer rate
        Q = min(max(Q, -1e6), 1e6)  # Limit to ±1 MW
        
        return Q, h 