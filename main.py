import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from background import create_background
from droplet_sizing import droplet_sizing
from dropsize_distribution import calculate_dropsize_distribution

# Settings
resolution = 9.86e-6  # Spatial resolution (m/pixel), adjust if needed
circularity_lower = 0.95
circularity_upper = 1.05

# Read images
root = 'Test_Images/'
image_files = [f for f in os.listdir(root) if f.lower().endswith(('.tif', 'png'))]
image_files.sort()

print(f"Number of files found: {len(image_files)}")

# Create composite background
background_images = [cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE) for f in image_files if cv2.imread(os.path.join(root, f)) is not None]

if not background_images:
    raise ValueError("No valid images found in the directory.")

background = create_background(np.array(background_images))

all_droplets = []
all_sharpness = []

for n, image_file in enumerate(image_files):
    print(f"\nProcessing image {n+1} of {len(image_files)}")
    
    img = cv2.imread(os.path.join(root, image_file), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to read image: {image_file}")
        continue
    
    img = img.astype(np.uint8)
    background = background.astype(np.uint8)
    
    img_corrected = cv2.divide(img, background, scale=255)
    img_corrected = cv2.normalize(img_corrected, None, 0, 255, cv2.NORM_MINMAX)

    centers, diameters, _, sharpness = droplet_sizing(img_corrected, resolution, circularity_lower, circularity_upper)
    
    if not sharpness:
        print(f"No droplets detected in image {n+1}")
        continue
    
    # Filter droplets based on sharpness (focus)
    sharpness_threshold = np.percentile(sharpness, 99)
    in_focus_indices = [i for i, s in enumerate(sharpness) if s >= sharpness_threshold]
    
    in_focus_centers = [centers[i] for i in in_focus_indices]
    in_focus_diameters = [diameters[i] for i in in_focus_indices]
    in_focus_sharpness = [sharpness[i] for i in in_focus_indices]
    
    all_droplets.extend(in_focus_diameters)
    all_sharpness.extend(in_focus_sharpness)
    
    if in_focus_diameters:
        avg_size = np.mean(in_focus_diameters) * 1e6  # Convert to micrometers
        size_range = (np.max(in_focus_diameters) - np.min(in_focus_diameters)) * 1e6
        variability = size_range / 2
        
        print(f"Number of in-focus droplets detected: {len(in_focus_diameters)}")
        print(f"Average droplet size: {avg_size:.2f} µm")
        print(f"Variability: {variability:.2f} µm")
        print("Droplet sizes (µm):")
        for size in in_focus_diameters:
            print(f"{size*1e6:.2f}")
    else:
        print(f"No in-focus droplets detected in image {n+1}")
    
    if n < 5 and in_focus_centers:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_corrected, cmap='gray')
        for center, diameter, sharp in zip(in_focus_centers, in_focus_diameters, in_focus_sharpness):
            circle = plt.Circle(center, diameter / 2 / resolution, color='r', fill=False, linewidth=1.5)
            plt.gca().add_artist(circle)
            plt.text(center[1], center[0], f'{sharp:.2f}', color='yellow', fontsize=8)
        plt.title(f"Image {n+1}: Detected In-Focus Droplets")
        plt.colorbar(label='Pixel intensity')
        plt.savefig(f'droplet_detection_{n+1}.png')
        plt.close()

if all_droplets:
    distribution = calculate_dropsize_distribution(all_droplets)

    print(f"\nOverall In-focus Droplet Size Distribution:")
    print(f"D_v10: {distribution['D_v10']:.2f} µm")
    print(f"D_v50: {distribution['D_v50']:.2f} µm")
    print(f"D_v90: {distribution['D_v90']:.2f} µm")
    print(f"Span: {distribution['span']:.2f}")
    print(f"Total number of in-focus droplets: {distribution['num_droplets']}")

    # Create histogram of droplet sizes
    plt.figure(figsize=(10, 6))
    plt.hist([d*1e6 for d in all_droplets], bins=30, edgecolor='black')
    plt.xlabel('Droplet Size (µm)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Droplet Sizes')
    plt.axvline(distribution['D_v10'], color='r', linestyle='dashed', linewidth=2, label='D_v10')
    plt.axvline(distribution['D_v50'], color='g', linestyle='dashed', linewidth=2, label='D_v50')
    plt.axvline(distribution['D_v90'], color='b', linestyle='dashed', linewidth=2, label='D_v90')
    plt.legend()
    plt.savefig('droplet_size_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(all_sharpness, bins=50)
    plt.xlabel('Edge Sharpness')
    plt.ylabel('Frequency')
    plt.title('Distribution of Edge Sharpness')
    plt.savefig('edge_sharpness_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter([d*1e6 for d in all_droplets], all_sharpness, alpha=0.5)
    plt.xlabel('Droplet Size (µm)')
    plt.ylabel('Edge Sharpness')
    plt.title('Droplet Size vs Edge Sharpness')
    plt.savefig('size_vs_sharpness.png')
    plt.close()

else:
    print("\nNo in-focus droplets detected in any of the images.")

print("\nFinished processing all images.")