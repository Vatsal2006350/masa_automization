import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from background import create_background
from droplet_sizing import droplet_sizing
from dropsize_distribution import calculate_dropsize_distribution

# Settings
resolution = 9.86e-6  # Spatial resolution (m/pixel)
max_diameter = 1500  # Maximum estimated spray diameter (µm)
circularity_lower = 0.7
circularity_upper = 1.5

# Read images
root = 'Test_Images/'
image_files = [f for f in os.listdir(root) if f.endswith('.tif')]
image_files.sort()

# Create composite background
background_images = []
for i in range(25):
    img = cv2.imread(os.path.join(root, image_files[i]), cv2.IMREAD_GRAYSCALE)
    background_images.append(img)

background = create_background(np.array(background_images))

print(f"Background shape: {background.shape}, dtype: {background.dtype}")

all_droplets = []

# Process each image
for n, image_file in enumerate(image_files):
    print(f"Processing image {n+1} of {len(image_files)}")

    # Read image
    img = cv2.imread(os.path.join(root, image_file), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image {image_file}")
        continue

    # Ensure data types
    img = img.astype(np.float32)
    background = background.astype(np.float32)

    # Background subtraction (normalize the image)
    img_corrected = cv2.divide(img, background, scale=255)

    # Convert back to uint8
    img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)

    # Droplet sizing
    centers, diameters, _ = droplet_sizing(img_corrected, resolution, circularity_lower, circularity_upper, focus_threshold=0.23)

    all_droplets.extend(diameters)

    print(f"Number of droplets detected: {len(diameters)}")
    print(f"Droplet sizes (µm): {[round(d*1e6, 2) for d in diameters[:10]]}...")  # Print first 10 droplet sizes

    # Only display results for the first 3 images
    if n < 3:
        plt.figure()
        plt.imshow(img_corrected, cmap='gray')
        for center, diameter in zip(centers, diameters):
            circle = plt.Circle((center[1], center[0]), diameter/2/resolution, color='r', fill=False)
            plt.gca().add_artist(circle)
        plt.title(f"Image {n+1}: Detected Droplets")
        plt.draw()
        plt.pause(0.001)
        plt.show()  # Show plot for first 3 images
    elif n == 3:
        print("Skipping display for remaining images to improve processing speed...")

# Calculate and display drop size distribution
if all_droplets:
    distribution = calculate_dropsize_distribution(all_droplets, max_diameter)

    print(f"\nOverall Droplet Size Distribution:")
    print(f"D_v10: {distribution['D_v10']:.2f} µm")
    print(f"D_v50: {distribution['D_v50']:.2f} µm")
    print(f"D_v90: {distribution['D_v90']:.2f} µm")
    print(f"Span: {distribution['span']:.2f}")
    print(f"Number of droplets: {distribution['num_droplets']}")
else:
    print("\nNo droplets detected in any of the images.")

print("\nFinished processing all images.")
