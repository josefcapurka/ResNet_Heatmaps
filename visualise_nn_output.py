import matplotlib.pyplot as plt
import random
import numpy as np

def visualise(image, nn_output, rows, output_filename):
    nn_output = nn_output.squeeze(0)
    print("Shape of network is {0}".format(nn_output.shape))
    blue_lane, yellow_lane = np.split(nn_output, 2)
    plt.clf()
    plt.plot(blue_lane, rows, marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
    plt.plot(yellow_lane, rows,  marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
    # plt.plot(x, r)
    plt.imshow(image)
    plt.savefig(output_filename + ".png")

