import cv2
import numpy as np
from mask import *

def perproccessing(img_path, output_path):
    
    image = cv2.imread(img_path)

    # Convert the image from BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel back with A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert the LAB image back to BGR color space
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    final = apply_circular_mask(final)

    # Save the image to the output folder
    cv2.imwrite(output_path, final)