import numpy as np
import cv2

def calculate_focus_parameter(img, dia, resolution):
    """
    Compute the focus parameter based on Lecuona et al. (2000)
    
    Args:
    img: Input image
    dia: Droplet diameter (m)
    resolution: Spatial resolution of the image (m/pixel)
    
    Returns:
    inf: Degree of focus of the droplet
    """
    # Compute gradients
    gv = cv2.Sobel(img.astype(float), cv2.CV_64F, 0, 1, ksize=3) / 8
    gh = cv2.Sobel(img.astype(float), cv2.CV_64F, 1, 0, ksize=3) / 8
    grad = np.sqrt(gv**2 + gh**2)
    
    # Sort gradient in descending order
    grad_sorted = np.sort(grad.flatten())[::-1]
    
    # Compute mean gradient at droplet edge
    peri = dia * np.pi / resolution
    a = max(0, int(round(peri / 2) - 3))
    b = min(len(grad_sorted), int(round(peri / 2) + 3))
    grad_max = np.mean(grad_sorted[a:b])
    
    # Compute background intensity
    img_sorted = np.sort(img.flatten())
    L = len(img_sorted)
    i_back = np.mean(img_sorted[int(0.80 * L):int(0.9 * L)])
    
    # Compute particle intensity (average of 5 lowest intensities)
    i_min = np.mean(img_sorted[:5])
    
    # Calculate focus parameter
    grad_comp = abs(i_min - i_back)
    inf = grad_max / grad_comp
    
    return inf