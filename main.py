import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image

images = [cv2.imread('1.png'), cv2.imread('2.png'), cv2.imread('3.png'), cv2.imread('4.png')]

for image in images:

    grey = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    # Blurring and erasing little details
    grey = cv2.GaussianBlur(grey, (9, 9), 0)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)

    # Thresholding to highlight the more dark areas
    #grey = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY)[1]
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)

    #canny = cv2.Canny(grey, 100, 200)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector

    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters

    params.filterByArea = True

    params.minArea = 10000
    params.maxArea = 100000

    # descobre qual a circularidade
    params.filterByCircularity = True

    circularity = 0

    for circularity in np.arange(1, 0, -0.01):
        params.minCircularity = circularity

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grey)

        if (len(keypoints) > 0):
            print("Circularity: ", circularity)
            params.filterByCircularity = False
            break

    #descoble a convexidade
    params.filterByConvexity = True

    convexity = 0

    for convexity in np.arange(1, 0, -0.01):
        params.minConvexity = convexity

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grey)

        if (len(keypoints) > 0):
            print("Convexity: ", convexity)
            params.filterByConvexity = False
            break

    #descoble a inercia
    params.filterByInertia = True

    inertia = 0

    for inertia in np.arange(1, 0, -0.01):
        params.minInertiaRatio = inertia

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grey)

        if (len(keypoints) > 0):
            print("inertia: ", inertia)
            params.filterByInertia = False
            break

    media = (circularity + convexity + inertia) / 3
    print("Média", media)
    # blank = np.zeros((1, 1))

    # blobs = cv2.drawKeypoints(grey, keypoints, blank, (255, 0, 0),

    #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # number_of_blobs = len(keypoints)

    # text = "Number of Circular Blobs: " + str(len(keypoints))

    # cv2.putText(blobs, text, (20, 550),

    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # # Show blobs

    # plt.imshow(blobs)
    # plt.show()
