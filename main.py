import os

import numpy as np
import cv2
import sys

from star_point import StarPoint
from analyze_images import *

width, height = 3024, 4032


def resize_one_image(w, h, image_path):
    image = cv2.imread(str(image_path))
    resized_image = cv2.resize(image, (w, h))
    cv2.imwrite("resized/" + str(image_path), resized_image)


def resize_images(w, h):
    for file_type in ["images"]:
        for img in os.listdir(file_type):
            image = cv2.imread(str(file_type) + "/" + str(img))
            resized_image = cv2.resize(image, (w, h))
            cv2.imwrite("resized/" + str(img), resized_image)


if len(sys.argv) != 2:
    """
    default images open
    """
    scan_image("fr1.jpg", True, True)
else:
    input_string = sys.argv[1].split(" ")
    if len(input_string) == 1:
        resize_one_image(width, height, input_string[0])
        scan_image(f"resized/{str(input_string[0])}", True, True)
    else:
        str_input = sys.argv[1].split(" ")
        resize_one_image(width, height, str_input[0])
        resize_one_image(width, height, str_input[1])
        out_img = run_matching_algorithm(f"resized/{str(str_input[0])}", f"resized/{str(str_input[1])}",
                                         f"{str(str_input[0].split('.')[0])}_{str(str_input[1].split('.')[0])}")

cv2.waitKey(0)
cv2.destroyAllWindows()

# show image to screen
# cv2.namedWindow('binary_image', cv2.WINDOW_NORMAL)
# cv2.imshow('binary_image', cv2.WINDOW_NORMAL)
