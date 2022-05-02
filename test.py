import cv2
import matplotlib.pyplot as plt
import numpy as np
 
# Load image

image = cv2.imread('1.png')

grey = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

kernel = np.ones((5, 5), np.uint8)
# Blurring and erasing little details
grey = cv2.GaussianBlur(grey, (9, 9), 0)
grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)

# Thresholding to highlight the more dark areas
grey = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY)[1]
grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)

canny = cv2.Canny(grey, 100, 200)
 
# Set our filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector

params = cv2.SimpleBlobDetector_Params()
 
# Set Area filtering parameters

params.filterByArea = False

params.minArea = 2000
 
# Set Circularity filtering parameters

params.filterByCircularity = True

params.minCircularity = 0.6
 
# Set Convexity filtering parameters

params.filterByConvexity = True

params.minConvexity = 0.1

     
# Set inertia filtering parameters

params.filterByInertia = True

params.minInertiaRatio = 0.01
 
# Create a detector with the parameters

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs

keypoints = detector.detect(canny)
 
# Draw blobs on our image as red circles

blank = np.zeros((1, 1)) 

blobs = cv2.drawKeypoints(canny, keypoints, blank, (0, 0, 255),

                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 

number_of_blobs = len(keypoints)

text = "Number of Circular Blobs: " + str(len(keypoints))

cv2.putText(blobs, text, (20, 550),

            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
 
# Show blobs

plt.imshow(blobs)
plt.show()