from time import sleep
import numpy as np
import cv2

img = cv2.imread('stars.jpg', cv2.IMREAD_GRAYSCALE)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary_img = cv2.threshold(img, 175, 235, cv2.THRESH_BINARY)
binary_img = cv2.resize(binary_img, (img.shape[1], img.shape[0]))
binary_img = cv2.imread(binary_img, cv2.)

gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

# binary_img = cv2.resize(binary_img, (img.shape[1], img.shape[0]))

print("finished")

# show image to screen
cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh', gray)

# wait for key to close
cv2.waitKey(0)

# circles = cv2.HoughCircles(image=binary_img, method=cv2.HOUGH_GRADIENT, dp=0.9,
#                            minDist=80, param1=110, param2=39, maxRadius=70)
# apply a blur using the median filter
# img = cv2.medianBlur(img, 5)

# Resize if needed:
# cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)

# img = cv2.GaussianBlur(img, (9, 9), 0)

# thresholding
# ret, binary_img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
#
# cv2.waitKey(0)
#
# circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 2, 3, param1=5, param2=10, minRadius=0, maxRadius=10)
#
# circles = np.uint16(np.around(circles))
# colored_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
#
# for i in circles[0, :]:
#     print(i)
#     # draw the outer circle
#     cv2.circle(colored_img, (i[0], i[1]), 12, (255, 255, 255), 2)
#     # draw the center of the circle
#     cv2.circle(colored_img, (i[0], i[1]), 1, (255, 255, 255), 3)
#
#     # calculate the brightness:
#     circle_center = (i[0], i[1])
#     circle_radius = i[2]
#
# # Create a binary mask for the circle
# mask = np.zeros_like(img, dtype=np.uint8)
# # rr, cc = skimage.draw.circle(circle_center[0], circle_center[1], circle_radius)
# np.where()
# mask[rr, cc] = 1
#
# # Extract the pixels inside the circle
# pixels_in_circle = img * mask
#
# # Calculate the average pixel value inside the circle
# average_pixel_value = np.mean(pixels_in_circle[np.nonzero(pixels_in_circle)])
#
# print(f"The average pixel value inside the circle is {average_pixel_value}")
