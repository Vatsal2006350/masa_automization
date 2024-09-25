import numpy as np

def track_droplets(C1, D1, C2, D2, image, resolution, diff_dia, angle, v_max, delay, inf):
    """
    Track droplets between two frames
    
    Args:
    C1, C2: Droplet centroid coordinates in first and second frame
    D1, D2: Droplet diameters in first and second frame
    image: Name of the processed frame
    resolution: Spatial resolution (m/pixel)
    diff_dia: Maximum relative difference in diameter accepted
    angle: Maximum angle between droplet trajectory and vertical (degrees)
    v_max: Maximum velocity in the spray (m/s)
    delay: Delay between two frames (s)
    inf: Focus parameter of each droplet
    
    Returns:
    droplets: List of tracked droplets
    rejected_droplets: List of droplets that couldn't be tracked
    """
    droplets = []
    rejected_droplets = []
    
    dep_max = v_max * delay / resolution  # Maximum allowed displacement
    angle_rad = angle * np.pi / 180  # Convert angle to radians
    
    if len(C1) > 0 and len(C2) > 0:
        for j in range(len(C1)):
            c1 = np.tile(C1[j], (len(C2), 1))
            dep = c1 - C2
            dx, dy = dep[:, 0], dep[:, 1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Find droplets fulfilling displacement criteria
            ind1 = np.where((np.abs(np.arctan2(np.abs(dx), np.abs(dy))) < angle_rad) & 
                            (dy < 0) & (dist < dep_max))[0]
            
            if len(ind1) > 0:
                # Apply size criteria
                d1 = np.full(len(ind1), D1[j])
                delta_dia = np.abs(d1 - D2[ind1]) / d1[0]
                ind2 = np.where(delta_dia < diff_dia)[0]
                ind = ind1[ind2]
                
                if len(ind) > 1:
                    ind = ind1[np.argmin(delta_dia[ind2])]
                
                if len(ind) == 1:
                    droplets.append({
                        'diameter': D1[j],
                        'diameter_avg': delta_dia[ind2][0],
                        'volume': ((D1[j]/2)**3) * np.pi * 4/3,
                        'displacement': dist[ind][0],
                        'center': np.vstack((C1[j], C2[ind])),
                        'velocity': dist[ind][0] * resolution / delay,
                        'image': image,
                        'focus': inf[j]
                    })
                    continue
            
            # If no match found, add to rejected droplets
            rejected_droplets.append({
                'diameter': D1[j],
                'center': C1[j],
                'image': image,
                'focus': inf[j]
            })
    
    return droplets, rejected_droplets