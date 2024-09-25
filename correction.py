import numpy as np

def calculate_correction_factor(velocity, diameter, delay, resolution):
    """
    Calculate correction factor for volumetric fractions.
    
    Args:
    velocity: Droplet velocity (m/s)
    diameter: Droplet diameter (m)
    delay: Time between two frames (s)
    resolution: Resolution of the imaging system (m/pixel)
    
    Returns:
    correction_factor: Correction factor for unequal sampling probability
    """
    FOV = ((1024 * resolution - (diameter / 2)) - (velocity * delay)) * ((1280 * resolution - (diameter / 2)))
    DOF = 0.85 * diameter + 0.00078
    P = (FOV * DOF) / velocity
    correction_factor = 1 / P
    return correction_factor