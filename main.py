from time import sleep
import numpy as np
import cv2

from StarPoint import StarPoint

original_img = cv2.imread('stars.jpg')
mask = np.zeros_like(original_img[:, :, 0])
img = cv2.imread('stars.jpg', cv2.IMREAD_GRAYSCALE)
(width, height) = img.shape

blackImage = np.ones((width, height), dtype=np.uint8)
ret, binary_img = cv2.threshold(img, 140, 235, cv2.THRESH_BINARY)
binary_img = cv2.blur(binary_img, (5, 5))

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
    cv2.circle(blackImage, (x, y), r + 10, (255, 255, 255), 2)
    # draw the number
    cv2.putText(blackImage, f"{star_id}", (x - 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0, 255), 2)
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
