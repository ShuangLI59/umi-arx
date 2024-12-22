import numpy as np
import matplotlib.pyplot as plt
import cv2
with open("data/experiments/1219/obs/24/16.npy", "rb") as f:
    obs = np.load(f, allow_pickle=True).item()

images = obs["obs_dict_np"]["camera0_rgb"]
print(images.shape)


# Ensure the images are in the correct format (N, 224, 224, 3)
images = np.transpose(images, (0, 2, 3, 1))

# Function to display an image
def display_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)

# Function to navigate through the images
def navigate_images(images):
    current_index = 0
    total_images = images.shape[0]

    while True:
        plt.clf()  # Clear the current plot
        print(f"current_index: {current_index}")
        display_image(images[current_index])
        print(f"Displaying image {current_index + 1}/{total_images}")

        key = input("Press 'n' for next, 'p' for previous, or 'q' to quit: ").strip().lower()

        if key == 'n':
            current_index = (current_index + 1) % total_images
        elif key == 'p':
            current_index = (current_index - 1) % total_images
        elif key == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid input. Use 'n', 'p', or 'q'.")

# Navigate through the images
navigate_images(images)
