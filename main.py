import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#img = cv.imread("1.png")
img = cv.imread("2.png")

grey = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

kernel = np.ones((5,5),np.uint8)
# Blurring and erasing little details
grey = cv.GaussianBlur(grey,(9,9),0)
grey = cv.morphologyEx(grey, cv.MORPH_OPEN, kernel)
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

#Thresholding to highlight the more dark areas
grey = cv.threshold(grey,140,255,cv.THRESH_BINARY)[1]
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

canny = cv.Canny(grey,100,200)
plt.imshow(canny,cmap="gray")
plt.show()