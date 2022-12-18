import cv2
import os

def crop_bounding_box(image_path, output_path):

    # Load the image
    im = cv2.imread(image_path)
    threshold, im2 = cv2.threshold(im, 10, 255, 0)

    # Use the Canny edge detector to find edges in the image
    edges = cv2.Canny(im2, 0, 255)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    c = max(contours, key = cv2.contourArea)

    # Create a bounding box around the contour
    x, y, w, h = cv2.boundingRect(c)

    # Crop the image to the square bounding box
    px = max(h,w)

    cropped_im = im[y:y + px, x:x + px]

    # Save the cropped image
    cv2.imwrite(output_path , cropped_im)
    
