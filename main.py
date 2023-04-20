from time import sleep
import numpy as np
import cv2

from StarPoint import StarPoint

original_img = cv2.imread('images/fr2.jpg')
mask = np.zeros_like(original_img[:, :, 0])
img = cv2.imread('images/fr2.jpg', cv2.IMREAD_GRAYSCALE)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
(width, height) = img.shape

blackImage = np.ones((width, height), dtype=np.uint8)
threshold = np.interp(np.average(img), [0, 255], [50, 400])
ret, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
binary_img = cv2.blur(binary_img, (7, 7), 0)

# cv2.namedWindow('binary_image', cv2.WINDOW_NORMAL)
# cv2.imshow('binary_image', binary_img)

stars = []
circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 8, 13, param1=5, param2=10, minRadius=0, maxRadius=10)

circles = np.uint16(np.around(circles))
star_id = 1

for (x, y, r) in circles[0, :]:
    cv2.circle(mask, (x, y), r, 1, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    brightness = np.mean(masked_img[mask == 1])
    stars.append(StarPoint(star_id, x, y, r, brightness))
    # draw the outer circle
    cv2.circle(blackImage, (x, y), r + 30, (255, 255, 255), 2)
    # draw the number
    cv2.putText(blackImage, f"{star_id}", (x - 40, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0, 255), 5)
    # draw the center of the circle
    cv2.circle(blackImage, (x, y), 1, (255, 255, 255), 3)
    star_id += 1

for star in stars:
    print(star)

# show image to screen
# cv2.namedWindow('binary_image', cv2.WINDOW_NORMAL)
# cv2.imshow('binary_image', cv2.WINDOW_NORMAL)
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow("original", original_img)
cv2.namedWindow('identified', cv2.WINDOW_NORMAL)
cv2.imshow('identified', blackImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# im = cv2.imread('images/fr1.jpg')
# blurred = cv2.GaussianBlur(im, (5, 5), 0)
# gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
# contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
#
# for contur in contours:
#     area = cv2.contourArea(contur)
#     print(area)
#
# cv2.namedWindow("mypic", cv2.WINDOW_NORMAL)
# cv2.imshow("mypic", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
