!pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

#uploaded = files.upload()

# Load a low-contrast grayscale image
image = cv2.imread('low contrast image.png', cv2.IMREAD_GRAYSCALE)

# 1. Histogram Equalization
hist_eq_image = cv2.equalizeHist(image)

# 2. CLAHE
# Create a CLAHE object with desired parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# Display the results
titles = ['Original Image', 'Histogram Equalization', 'CLAHE']
images = [image, hist_eq_image, clahe_image]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
