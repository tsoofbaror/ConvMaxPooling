import cv2
import numpy as np

def apply_convolution(image, kernel):
    # Apply convolution
    convolved_image = cv2.filter2D(image, -1, kernel)
    return convolved_image

def apply_max_pooling(image, k):
    # Apply max pooling
    pooled_image = cv2.resize(image, (image.shape[1] // k, image.shape[0] // k), interpolation=cv2.INTER_NEAREST)
    return pooled_image

def save_image(image, filename):
    cv2.imwrite(filename, image)

# Load the image
image = cv2.imread("input_image.jpeg", cv2.IMREAD_GRAYSCALE)
print("Original Image Size:", image.shape)

#Create the edge detecting kernel
kernel_size = 3
kernel = np.zeros((kernel_size, kernel_size))
kernel[0, :] = 1
kernel[-1, :] = -1
print("Filter:")
print(kernel)


# Perform convolution
convolved_image = apply_convolution(image, kernel)

# Perform max pooling
k = 4
pooled_image = apply_max_pooling(convolved_image, k)

# Save the images
save_image(convolved_image, "convolved_image.jpg")
save_image(pooled_image, "pooled_image.jpg")
