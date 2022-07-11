import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):

    print("tested image")
    grey = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    grey = cv.equalizeHist(grey)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    grey = cv.medianBlur(grey, 15)
    grey = cv.GaussianBlur(grey, (15, 15), 0)

    grey = cv.threshold(grey, 140, 255, cv.THRESH_TOZERO)[1]

    grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)
    
    grey = cv.dilate(grey , kernel2 ,iterations = 20)
    grey = cv.erode(grey , kernel2,iterations = 21)
 
    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True

    params.minArea = 10000
    params.maxArea = 100000


    # descobre a convexidade
    params.filterByConvexity = True

    convexity = 0

    for convexity in np.arange(1, 0, -0.01):
        params.minConvexity = convexity

        detector = cv.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grey)

        if (len(keypoints) > 0):
            print("Convexity: ", convexity)
            params.filterByConvexity = False
            break

    # descobre a inercia
    params.filterByInertia = True

    inertia = 0

    for inertia in np.arange(1, 0, -0.01):
        params.minInertiaRatio = inertia

        detector = cv.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grey)

        if (len(keypoints) > 0):
            print("inertia: ", inertia)
            params.filterByInertia = False
            break

    media = (convexity + inertia) / 2

    blank = np.zeros((1, 1))

    blobs = cv.drawKeypoints(grey, keypoints, blank, (255, 0, 0),

                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    blobs_len = len(keypoints)

    return {"convexity": convexity, "inertia": inertia, "media": media, "blobs": blobs_len}

