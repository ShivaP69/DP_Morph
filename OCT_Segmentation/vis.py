import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Used here just for the addWeighted function

def layers_plot(mask_np):

# Assuming layer values are known or documented
# For example, let's say we have layer values 1, 2, 3, etc.
    layers = [1, 2, 3,4,5,6,7,8,9]  # Update this list based on your specific layer identifiers

    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(15, 5))

    # Plot the original mask
    axes[0].imshow(mask_np, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')

    # Plot each layer
    for i, layer in enumerate(layers):
        layer_mask = np.where(mask_np == layer, 1, 0)
        axes[i + 1].imshow(layer_mask, cmap='gray')
        axes[i + 1].set_title(f'Layer {layer}')
        axes[i + 1].axis('off')

    plt.show()




def layers_plot_with_image(image_np, mask_np):
    """
    Plots the original image with different mask layers overlaid on top.
    """
    # Assuming layer values are known or documented
    # For example, let's say we have layer values 1, 2, 3, etc.
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Update this list based on your specific layer identifiers

    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(20, 10))

    # Normalize the image for better visualization if it's not in the 0-255 range
    if image_np.max() > 1.0:
        image_normalized = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        image_normalized = (image_np * 255).astype(np.uint8)

    # Convert to a 3-channel image if it's grayscale
    if len(image_normalized.shape) == 2:
        image_normalized = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)

    # Plot the original image
    axes[0].imshow(image_normalized, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot each layer overlaid on the image
    for i, layer in enumerate(layers):
        # Create a binary mask for the current layer
        layer_mask = np.where(mask_np == layer, 255, 0).astype(np.uint8)

        # Convert the mask to 3-channel to match the image
        layer_mask_color = cv2.applyColorMap(layer_mask, cv2.COLORMAP_JET)

        # Overlay the mask on the image
        overlay = cv2.addWeighted(image_normalized, 0.7, layer_mask_color, 0.3, 0)

        # Display the overlay
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f'Layer {layer}')
        axes[i + 1].axis('off')

    plt.show()

# Load an example image and mask from .npy files


dataset='Duke'
data_path= '../DukeData'
num_classes= 2



print("Current Directory:", os.getcwd())
base_dir = f'/home/parsar0000/oct_git/main_code/{data_path}/test'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')

for image_filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_filename)
    # mask_filename = image_filename.replace('.jpg', '_mask.jpg')
    mask_filename = image_filename
    mask_path = os.path.join(mask_dir, mask_filename)

    # Call the function to plot the layers with the image
    layers_plot_with_image(image_path, mask_path)
