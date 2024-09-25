import numpy as np

def create_background(images):
    """
    Create a composite background from a set of images.
    
    Args:
    images: 3D numpy array of images (n_images, height, width)
    
    Returns:
    background: 2D numpy array representing the composite background
    """
    # Sort pixels by intensity along the first axis (across images)
    sorted_images = np.sort(images, axis=0)
    
    # Create composite background (80th percentile)
    background = sorted_images[int(images.shape[0] * 0.8)]
    
    return background