import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, segmentation
from focus import calculate_focus_parameter  # Import focus calculation function

def droplet_sizing(image, resolution, circ_inf, circ_up, focus_threshold=0.23):
    """
    Detect and measure droplets in the image, considering only those in focus.

    Args:
    image: Input image
    resolution: Spatial resolution of the image (m/pixel)
    circ_inf, circ_up: Minimum and maximum allowed circularity
    focus_threshold: Minimum focus parameter for droplets to be accepted

    Returns:
    centers: Centroid positions of droplets
    diameters: Diameters of droplets
    circularity: Circularity of droplets
    """
    # Ensure image is float32 for precision in gradient computation
    image = image.astype(np.float32)

    # Computation of the gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    Grad = cv2.magnitude(sobelx, sobely)

    # Normalize gradient to range 0-255
    Grad_norm = cv2.normalize(Grad, None, 0, 255, cv2.NORM_MINMAX)
    Grad_norm = Grad_norm.astype(np.uint8)

    # Apply threshold using Otsu's method
    _, Grad_thresh = cv2.threshold(Grad_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to binary image
    Grad_bin = Grad_thresh

    # Fill holes
    Grad_filled = ndimage.binary_fill_holes(Grad_bin).astype(np.uint8)

    # Morphological operations
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    Grad_eroded = cv2.erode(Grad_filled, se)
    Grad_dilated = cv2.dilate(Grad_eroded, se)
    Grad_cleared = segmentation.clear_border(Grad_dilated).astype(bool)

    # Label connected regions
    labeled_droplets = measure.label(Grad_cleared)
    props = measure.regionprops(labeled_droplets)

    centers = []
    diameters = []
    circularity = []

    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Reject droplets touching the image border
        if prop.bbox[0] == 0 or prop.bbox[1] == 0 or prop.bbox[2] == image.shape[0] or prop.bbox[3] == image.shape[1]:
            continue

        # Reject droplets smaller than 5 pixels in diameter
        diameter_pixels = 2 * np.sqrt(area / np.pi)
        if diameter_pixels < 5:
            continue

        # Calculate the droplet diameter in meters
        diameter_m = diameter_pixels * resolution

        # Compute the focus parameter
        focus_value = calculate_focus_parameter(image, diameter_m, resolution)

        # Reject droplets not in focus
        if focus_value < focus_threshold:
            continue

        if circ_inf <= circ <= circ_up:
            centers.append(prop.centroid)
            diameters.append(diameter_m)  # Already in meters
            circularity.append(circ)

    return centers, diameters, circularity
