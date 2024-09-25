import numpy as np
import matplotlib.pyplot as plt

def plot_displacement(droplets, background_image):
    """
    Plot droplet displacements on the background image.
    
    Args:
    droplets: List of tracked droplets
    background_image: Background image to plot on
    """
    plt.figure()
    plt.imshow(background_image, cmap='gray')
    
    for droplet in droplets:
        # Draw the droplet (circle with same diameter)
        radius = round(droplet['diameter'][0] / (2 * 9.6e-6))
        center1 = np.round(droplet['center'][0]).astype(int)
        theta = np.linspace(0, 2*np.pi, 50)
        x = radius * np.cos(theta) + center1[0]
        y = radius * np.sin(theta) + center1[1]
        plt.plot(x, y, '-r', linewidth=2)
        
        # Draw the displacement
        center2 = np.round(droplet['center'][1]).astype(int)
        plt.plot([center1[0], center2[0]], [center1[1], center2[1]], 'k-')
        plt.plot([center2[0]-5, center2[0]], [center2[2]-5, center2[1]], 'k-')
        plt.plot([center2[0]+5, center2[0]], [center2[2]-5, center2[1]], 'k-')
    
    plt.draw()
    plt.pause(0.001)