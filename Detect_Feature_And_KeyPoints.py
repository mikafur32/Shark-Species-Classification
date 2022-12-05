import cv2
import numpy as np

def Detect_Feature_And_KeyPoints(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # detect and extract features from the image
  orb = cv2.ORB_create()
  (Keypoints, features) = orb.detectAndCompute(gray, None)

  Keypoints = np.float32([i.pt for i in Keypoints])
  return (Keypoints, features)