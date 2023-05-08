import numpy as np
from PIL import Image


def add_periodic_noise(
    image_path, output_path, noise_type="vertical", amplitude=30, frequency=5
):
    # convert to gray scale
    image = Image.open(image_path).convert("L")
    # resize to 256x256
    image = image.resize((256, 256))
    width, height = image.size
    # make sure it won't overflow
    img_array = np.array(image, dtype=int)

    if noise_type == "vertical":
        for x in range(width):
            for y in range(height):
                img_array[y, x] += int(
                    amplitude * np.sin(2 * np.pi * frequency * x / width)
                )
    elif noise_type == "horizontal":
        for x in range(width):
            for y in range(height):
                img_array[y, x] += int(
                    amplitude * np.sin(2 * np.pi * frequency * y / height)
                )
    elif noise_type == "spear":
        for x in range(width):
            for y in range(height):
                img_array[y, x] += int(
                    amplitude
                    * np.sin(2 * np.pi * frequency * (x + y) / (width + height))
                )

    # clip to 0-255
    img_array = np.clip(img_array, 0, 255)
    noisy_image = Image.fromarray(np.uint8(img_array))
    noisy_image.save(output_path)


# Example usage:
image_path = "figures/2.png"
output_path = "figures/2_noisy.png"
noise_type = "spear"  # Choose between "vertical", "horizontal", or "spear"
amplitude = 100
frequency = 20

add_periodic_noise(image_path, output_path, noise_type, amplitude, frequency)
