import cv2
import numpy as np


def apply_circular_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(gray_image)
    
    height, width = gray_image.shape[:2]
    center = (width // 2, height // 2)
    radius = max(center[1], center[1])  # Make the circle fit within the image dimensions
    
    cv2.circle(mask, center, radius, (255), thickness=-1)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image