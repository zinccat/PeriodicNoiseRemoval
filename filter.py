import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def apply_laplacian_kernel(image):
    laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    boundary_image = cv2.filter2D(image, -1, laplacian_kernel)
    return boundary_image


def show_frequency(image):
    out = np.log(1 + np.abs(image))
    plt.imshow(out, cmap="gray")
    # plt.axis("off")
    return out


def frequency(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift


def filter(input_path, output_path):
    image = Image.open(input_path)
    image = np.array(image)
    freq = frequency(image)
    # filtered
    psd_filtered = np.zeros(freq.shape, dtype=complex)
    window_size = 1
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            mean = np.mean(
                freq[
                    i - window_size : i + window_size + 1,
                    j - window_size : j + window_size + 1,
                ]
            )
            if np.abs(freq[i, j]) > 8 * np.abs(mean):
                psd_filtered[i, j] = mean
            else:
                psd_filtered[i, j] = freq[i, j]
    image_filtered = np.fft.ifft2(np.fft.ifftshift(psd_filtered))
    image_filtered = np.real(image_filtered)
    plt.imshow(image_filtered, cmap="gray")
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    filter("figures/1_noisy.png", "figures/1_filtered.png")
