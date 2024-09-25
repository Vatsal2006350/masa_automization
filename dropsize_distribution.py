import numpy as np
import matplotlib.pyplot as plt
from correction import calculate_correction_factor

def calculate_dropsize_distribution(droplets, delay, resolution, max_diameter):
    """
    Calculate and plot drop size distribution.
    
    Args:
    droplets: List of tracked droplets
    delay: Time between two frames (s)
    resolution: Resolution of the imaging system (m/pixel)
    max_diameter: Maximum estimated spray diameter (µm)
    
    Returns:
    Dictionary containing distribution parameters
    """
    diameters = np.array([d['diameter'] for d in droplets])
    volumes = np.array([d['volume'] for d in droplets])
    velocities = np.array([d['velocity'] for d in droplets])
    
    # Calculate correction factors
    correction_factors = calculate_correction_factor(velocities, diameters, delay, resolution)
    
    # Apply correction to volumes
    corrected_volumes = volumes * correction_factors
    
    # Sort by diameter
    sorted_indices = np.argsort(diameters)
    diameters_sorted = diameters[sorted_indices] * 1e6  # Convert to µm
    volumes_sorted = corrected_volumes[sorted_indices]
    
    cumulative_volume = np.cumsum(volumes_sorted)
    cumulative_volume_percent = 100 * cumulative_volume / np.sum(volumes_sorted)
    
    # Calculate distribution parameters
    d_v10 = np.interp(10, cumulative_volume_percent, diameters_sorted)
    d_v50 = np.interp(50, cumulative_volume_percent, diameters_sorted)
    d_v90 = np.interp(90, cumulative_volume_percent, diameters_sorted)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Velocity vs Diameter plot
    ax1.loglog(diameters * 1e6, velocities, '*')
    ax1.set_xlabel('Diameter [µm]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_xlim(0, 3000)
    ax1.set_ylim(0.1, 15)
    
    # Drop size distribution plot
    bins = np.arange(0, max_diameter + 50, 50)
    hist, _ = np.histogram(diameters_sorted, bins=bins, weights=volumes_sorted)
    
    ax2.bar(bins[:-1], 100 * hist / np.sum(hist), width=50, alpha=0.7)
    ax2.set_xlabel('Diameter [µm]')
    ax2.set_ylabel('Relative volume [%]')
    ax2.set_xlim(0, max_diameter)
    ax2.set_ylim(0, 25)
    
    ax3 = ax2.twinx()
    ax3.plot(diameters_sorted, cumulative_volume_percent, 'r-', linewidth=2)
    ax3.set_ylabel('Cumulative relative volume [%]')
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'D_v10': d_v10,
        'D_v50': d_v50,
        'D_v90': d_v90,
        'span': (d_v90 - d_v10) / d_v50,
        'num_droplets': len(droplets),
        'total_volume': np.sum(volumes),
        'corrected_total_volume': np.sum(corrected_volumes)
    }