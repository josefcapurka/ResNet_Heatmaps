import matplotlib.pyplot as plt
import random
import numpy as np
import cv2 as cv

def visualise(image, blue_lane, yellow_lane, rows, output_filename):
    plt.clf()
    plt.plot(blue_lane, rows, marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
    # print(blue_lane.numpy())
    plt.plot(yellow_lane, rows,  marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
    # plt.plot(x, r)
    blue_pts = np.column_stack((blue_lane, rows))
    yellow_pts = np.column_stack((yellow_lane, rows))
    poly_pts = np.vstack((blue_pts, yellow_pts[::-1]))
    poly_pts = np.vstack((poly_pts, np.array([blue_pts[0]]))).astype(int)
    # print(poly_pts)
    # print(output_filename)
    plt.imshow(image)
    plt.savefig(output_filename)
    # plt.savefig(output_filename + ".png")
    # poly_pts = poly_pts * 2
    zeros = np.zeros((420, 1280))
    # cv.fillPoly(zeros, pts=[poly_pts], color=(255))
    # cv.imwrite(output_filename, zeros)
