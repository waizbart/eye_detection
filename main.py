import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("1.png")
#img = cv.imread("2.png")

grey = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

kernel = np.ones((5, 5), np.uint8)
# Blurring and erasing little details
grey = cv.GaussianBlur(grey, (9, 9), 0)
grey = cv.morphologyEx(grey, cv.MORPH_OPEN, kernel)
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

# Thresholding to highlight the more dark areas
grey = cv.threshold(grey, 140, 255, cv.THRESH_BINARY)[1]
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

canny = cv.Canny(grey, 100, 200)

circles = cv.HoughCircles(grey,
                          cv.HOUGH_GRADIENT,
                          dp=1.1,
                          minDist=300,
                          param1=200,
                          param2=40,
                          minRadius=50,
                          maxRadius=400)

circles = np.uint16(np.around(circles))
cimg = img.copy()
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(255,0,0),10)
    
plt.imshow(cimg)
plt.show()
