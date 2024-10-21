import os
import numpy as np
from PIL import Image
from models.funie_gan import FUNIE_GAN

# Path to the directory containing new images
IMAGE_DIR = 'data/raw/'

# Define the function to enhance a single image
def enhance_image(image_path):
    # Load the trained FUNIE-GAN model
    model = FUNIE_GAN()
    model.generator.load_weights('checkpoints/generator.h5')

    # Load and preprocess the input image
    image = np.array(Image.open(image_path).resize((256, 256)))
    image = (image / 127.5) - 1  # Normalize image to [-1, 1]
    
    # Use the model to enhance the image
    enhanced_image = model.generator(np.expand_dims(image, axis=0), training=False)
    print(enhanced_image)
    # Post-process the enhanced image (convert back to [0, 255])
    enhanced_image = ((enhanced_image[0] + 1) * 127.5).numpy().astype(np.uint8)

    # Save the enhanced image
    Image.fromarray(enhanced_image).save(f'results/enhanced_{os.path.basename(image_path)}')

# Enhance all images in the directory
def enhance_images_in_directory(directory_path):
    # Loop through each image in the directory
    for image_name in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_name)
        if image_name.endswith(".jpg") or image_name.endswith(".png"):  # Make sure it's an image file
            enhance_image(image_path)  # Call the enhance_image function

if __name__ == "__main__":
    enhance_images_in_directory(IMAGE_DIR)
