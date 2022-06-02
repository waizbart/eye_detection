import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mahotas

eyes = [cv.imread(str(i) + '.png') for i in range(1, 14)]

for image in eyes:
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

    text = "Circularidade: " + str(round(media, 4))

    #blobs image size
    height, width = blobs.shape[:2]

    textX = int(width/10)
    textY = int(height/10)
    
    cv.putText(blobs, text, (textX, textY),
                cv.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 2)

    # Show blobs

    plt.imshow(blobs)
    plt.show()

