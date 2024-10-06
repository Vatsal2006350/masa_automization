import numpy as np
import matplotlib.pyplot as plt

def calculate_dropsize_distribution(diameters):
    """
    Calculate and plot drop size distribution.
    
    Args:
    diameters: List of droplet diameters (m)
    
    Returns:
    Dictionary containing distribution parameters
    """
    diameters_um = np.array(diameters) * 1e6  # Convert to µm
    
    # Sort diameters
    diameters_sorted = np.sort(diameters_um)
    
    # Calculate cumulative volume
    volumes = (4/3) * np.pi * (diameters_sorted/2)**3
    cumulative_volume = np.cumsum(volumes)
    cumulative_volume_percent = 100 * cumulative_volume / np.sum(volumes)
    
    # Calculate distribution parameters
    d_v10 = np.interp(10, cumulative_volume_percent, diameters_sorted)
    d_v50 = np.interp(50, cumulative_volume_percent, diameters_sorted)
    d_v90 = np.interp(90, cumulative_volume_percent, diameters_sorted)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(diameters_um, bins=50, weights=volumes, density=True, alpha=0.7)
    plt.xlabel('Diameter [µm]')
    plt.ylabel('Volume Fraction')
    
    # Cumulative volume curve
    ax2 = plt.twinx()
    ax2.plot(diameters_sorted, cumulative_volume_percent, 'r-', linewidth=2)
    ax2.set_ylabel('Cumulative Volume [%]')
    ax2.set_ylim(0, 100)
    
    plt.title('Droplet Size Distribution')
    plt.xlim(0, max(diameters_um))
    
    return {
        'D_v10': d_v10,
        'D_v50': d_v50,
        'D_v90': d_v90,
        'span': (d_v90 - d_v10) / d_v50,
        'num_droplets': len(diameters)
    }