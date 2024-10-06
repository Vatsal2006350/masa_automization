import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, segmentation, filters

def edge_sharpness(image, mask):
    """
    Compute the edge sharpness of a droplet based on the abruptness of color change.
    """
    # Dilate and erode the mask to get the edge region
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    edge_region = dilated - eroded

    # Compute gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Compute mean gradient magnitude along the edge
    edge_sharpness = np.mean(grad_mag[edge_region == 1])
    
    return edge_sharpness

def droplet_sizing(image, resolution, circ_inf, circ_up):
    """
    Detect and measure droplets in the image.
    
    Args:
    image: Input image
    resolution: Spatial resolution of the image (m/pixel)
    circ_inf, circ_up: Minimum and maximum allowed circularity
    
    Returns:
    centers: Centroid positions of droplets
    diameters: Diameters of droplets
    circularity: Circularity of droplets
    sharpness: Edge sharpness of droplets
    """
    # Edge detection
    edges = filters.sobel(image)
    
    # Threshold to create binary image
    thresh = filters.threshold_otsu(edges)
    binary = edges > thresh

    # Remove small objects and fill holes
    binary = ndimage.binary_opening(binary)
    binary = ndimage.binary_closing(binary)

    # Label connected components
    labeled = measure.label(binary)

    # Measure properties of labeled image regions
    props = measure.regionprops(labeled, image)

    centers = []
    diameters = []
    circularity = []
    sharpness = []

    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        circ = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        if circ_inf <= circ <= circ_up:
            # Extract region around the droplet
            bbox = prop.bbox
            roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            mask = labeled[bbox[0]:bbox[2], bbox[1]:bbox[3]] == prop.label
            
            # Compute edge sharpness
            es = edge_sharpness(roi, mask)
            
            centers.append(prop.centroid)
            diameter = 2 * np.sqrt(area / np.pi) * resolution  # Diameter in meters
            diameters.append(diameter)
            circularity.append(circ)
            sharpness.append(es)

    return centers, diameters, circularity, sharpness