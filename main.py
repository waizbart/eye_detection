import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

eyes = [cv.imread(str(i) + '.png') for i in range(1, 14)]

for image in eyes:
    grey = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    # Blurring and erasing little details
    grey = cv.GaussianBlur(grey, (9, 9), 0)
    grey = cv.morphologyEx(grey, cv.MORPH_OPEN, kernel)
    grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

    # Thresholding to highlight the more dark areas
    grey = cv.threshold(grey, 150, 255, cv.THRESH_BINARY)[1]
    grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

    #canny = cv.Canny(grey, 100, 200)

    # Set our filtering parameters
    # Initialize parameter setting using cv.SimpleBlobDetector

    params = cv.SimpleBlobDetector_Params()

    # Set Area filtering parameters

    params.filterByArea = True

    params.minArea = 10000
    params.maxArea = 100000

    # descobre qual a circularidade
    params.filterByCircularity = True

    circularity = 0

    for circularity in np.arange(1, 0, -0.01):
        params.minCircularity = circularity

        detector = cv.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grey)

        if (len(keypoints) > 0):
            print("Circularity: ", circularity)
            params.filterByCircularity = False
            break

    # descoble a convexidade
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

    # descoble a inercia
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

    media = (circularity + convexity + inertia) / 3
    print("Média:", media, "\n")
    blank = np.zeros((1, 1))

    blobs = cv.drawKeypoints(grey, keypoints, blank, (255, 0, 0),

                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)

    text = "Number of Circular Blobs: " + str(len(keypoints))

    cv.putText(blobs, text, (20, 550),

                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs

    plt.imshow(blobs)
    plt.pause(3)
    
plt.show()
