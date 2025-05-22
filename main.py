import cv2
import numpy as np

img = cv2.imread(cv2.samples.findFile("immagini\\new.jpg"))
hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 150], np.uint8)
green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areaContorni=[]
if contours:
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        areaContorni.append(w*h)
    x, y, w, h = cv2.boundingRect(contours[areaContorni.index(max(areaContorni))])
    img = cv2.rectangle(img, (x, y),
                            (x + w, y + h),
                            (0, 255, 0), 2)
cv2.imshow("Display window", img)
k = cv2.waitKey(0)