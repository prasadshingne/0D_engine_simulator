import numpy as np
from engine_types import OperatingConditions, InitialConditions, GeometryParameters

def _calculate_piston_position(theta: float, geom: GeometryParameters) -> float:
    """
    Calculate instantaneous piston position.
    
    Args:
        theta: Crank angle [rad]
        geom: Geometry parameters
    
    Returns:
        Piston position [m]
    """
    return (geom.a * np.cos(theta) + 
            np.sqrt(geom.conr**2 - (geom.a * np.sin(theta))**2))

def area1(t: float, init: InitialConditions, geom: GeometryParameters, oper: OperatingConditions) -> float:
    """
    Calculate instantaneous cylinder area based on engine geometry and operating parameters.
    
    Parameters:
    -----------
    t : float
        Time [s]
    init : InitialConditions
        Initial conditions containing theta (initial crank angle)
    geom : GeometryParameters
        Engine geometry parameters
    oper : OperatingConditions
        Engine operating conditions
    
    Returns:
    --------
    float
        Total instantaneous cylinder area [m²]
    """
    # Calculate current crank angle
    theta = init.theta + oper.rpm * 2 * np.pi / 60 * t
    
    # Update piston position
    geom.x = _calculate_piston_position(theta, geom)
    
    # Calculate cylinder head area (constant)
    head_area = np.pi / 4 * geom.bore**2
    
    # Calculate instantaneous cylinder wall area
    wall_area = np.pi * geom.bore * (geom.conr + geom.a - geom.x)
    
    return head_area + wall_area 