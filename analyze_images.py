import cv2
import numpy as np
import networkx as nx
import math
import csv

import star_point
from star_point import StarPoint

width = 0
height = 0
first = True

header = ['id', 'x', 'y', 'radius', 'brightness']


def write_image_to_outputs(img, name):
    cv2.imwrite(f"outputs/{name}.png", img)


def write_to_csv_file(stars, file_name):
    f = open(f"outputs/{str(file_name)}.csv", 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    for star in stars:
        row = [str(star.get_id()), str(star.get_x()), str(star.get_y()), str(star.get_radius()),
               str(star.get_brightness())]
        writer.writerow(row)


def load_image_and_mask(path: str):
    """
    Open images as cv2 with grayscale color
    :param path:
    :return:
    """
    global first
    global width
    global height
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(img[:, :])
        return img, mask
    except IOError:
        raise IOError("Something went wrong with file path")


def show(img, window_name):
    cv2.namedWindow(f"{window_name}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{window_name}", img)


def valid_name(path):
    name = ""
    if '/' in path:
        name = path.split('/')[1]
        if '.' in name:
            name.split('.')
    else:
        name = path.split('.')
    return name


def scan_image(path: str, show_images=False, print_star=False):
    name = valid_name(path)
    img, mask = load_image_and_mask(path)
    black_img = np.ones(img.shape, dtype=np.uint8)

    threshold = np.interp(np.average(img), [0, 255], [50, 400])
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    stars = []
    star_id = 1

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(max(w, h) / 2)
        cv2.circle(mask, center, radius, 1, -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        brightness = np.mean(masked_img[mask == 1])
        stars.append(StarPoint(star_id, center[0], center[1], radius, brightness))
        # draw the outer circle
        cv2.circle(black_img, center, radius + 30, (255, 255, 255), 4)
        # draw the number
        cv2.putText(black_img, f"{star_id}", (center[0] - 20, center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 0, 255), 3)
        # draw the center of the circle
        cv2.circle(black_img, center, 1, (255, 255, 255), 2)
        star_id += 1
    if print_star:
        for star in stars:
            print(star)
    concatenated = cv2.hconcat((img, black_img))
    if show_images:
        show(concatenated, "orig")
    # G = build_graph(stars)
    write_to_csv_file(stars, name)
    write_image_to_outputs(concatenated, f"{name}_concatenated")
    return img, black_img


def calc_dist(star: StarPoint, neigh: StarPoint) -> float:
    p = [star.get_x(), star.get_y()]
    q = [neigh.get_x(), neigh.get_y()]
    return math.dist(p, q)


def build_graph(stars_object):
    G = nx.Graph()
    for star in stars_object:
        G.add_node(star)
    for star_node in stars_object:
        for neigh_node in stars_object:
            if star_node.__eq__(neigh_node) or G.has_edge(star_node, neigh_node):
                continue
            else:
                dist = calc_dist(star_node, neigh_node)
                G.add_edge(star_node, neigh_node, weight=dist)
                G.add_edge(neigh_node, star_node, weight=dist)
    return G


def run_matching_algorithm(img1_path, img2_path, name):
    """
    Tried to use this matching algorithm to find at least two stars that are matching in both images
    and with this two stars to evaluate the ratio of the image, but no matter what a matching algorithm we tried to use
    we didn't get any good one that will fit and find the accurate matching.
    :param img1_path:
    :param img2_path:
    :return:
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # BFMatcher with default params

    matches = flann.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    good = sorted(good, key=lambda x: x[0].distance)
    top_good = good[:1000]
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, top_good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    w, h = img3.shape[:2]
    cv2.line(img3, (int(h / 2), 0), (int(h / 2), w), color=(0, 255, 0), thickness=3)
    write_image_to_outputs(img3, name=f"{name}_matched")
    show(img3, "Matched")
