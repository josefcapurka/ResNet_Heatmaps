
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import json
import jsonlines
from pathlib import Path
import random
# from dataset import fsDataset
num_segments_per_line = 20
rows = np.linspace(419, 50, num_segments_per_line).astype(int)


def get_mask(label_path):
        height = 420
        width = 1280
        left_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the lines on the mask with white color (255)
        thickness = 1
        color = 255
        # single coord format is: start_point.x, start_point.y, end_point.x, end_point.y
        with open(label_path, 'r') as f:
            data = json.load(f)
            left_coords = data["left_coords"]
            right_coords = data["right_coords"]
            # plt.imshow(img_arr)
            for left_segment in left_coords:
                line_start = (int(left_segment[0]), int(left_segment[1]))
                line_end = (int(left_segment[2]), int(left_segment[3]))
                # if left_segment[3] > left_segment[1]:
                #     # print("detected")
                #     continue
                # print(left_segment[1], left_segment[3])
                cv2.line(left_mask, line_start, line_end, color, thickness)

            blue_lane = np.where(left_mask == 255)
            _, left_indices = np.unique(blue_lane[0], return_index=True)
            l_indices = (blue_lane[0][left_indices], blue_lane[1][left_indices])
            # blue_lane[1][left_indices]
            # print(indices)
            right_mask = np.zeros((height, width), dtype=np.uint8)

                # cv2.line(mask, line2_start, line2_end, color, thickness)
                # plt.plot([left_segment[0], left_segment[2]], [left_segment[1], left_segment[3]], color='b')
            for right_segment in right_coords:
                line_start = (int(right_segment[0]), int(right_segment[1]))
                line_end = (int(right_segment[2]), int(right_segment[3]))
                # if right_segment[3] > right_segment[1]:
                #     # print("detected")
                #     continue
                cv2.line(right_mask, line_start, line_end, color, thickness)
            yellow_lane = np.where(right_mask == 255)
            _, right_indices = np.unique(yellow_lane[0], return_index=True)
            r_indices = (yellow_lane[0][right_indices], yellow_lane[1][right_indices])
            # print(f"image name is: {str(label_path).split('/')[-1]}")
            left_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask = np.zeros((height, width), dtype=np.uint8)
            left_mask[l_indices] = 255
          
            left_mask[l_indices] = 255
            right_mask[r_indices] = 255

            # cv2.imwrite("visualise_labels/" + str(label_path).split("/")[-1] + ".png", left_mask + right_mask)
        mask = {}
        left_mask[~rows] = 0
        right_mask[~rows] = 0
        # print(rows)
        mask["left_lane"] = left_mask #[rows]
        mask["right_lane"] = right_mask #[rows]

        return mask

def get_synt_mask_measurements(img_name):
        img_name = img_name.split("/")[-1]
        # example of img_name: run_261_000008
        # print(img_name)
        run_number = img_name.split("_")[1]
        # print(f"run number is {run_number}")
        img_id = img_name.split("_")[2][0:-4]
        dir_mask = Path('/home/capurjos/synt_data/labels')
        jsonl_path = os.path.join(dir_mask, "labels_run_" + run_number + ".jsonl")
        # print(f"path to json is {jsonl_path}")
        height = 720
        width = 1280
        left_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the lines on the mask with white color (255)
        thickness = 1
        color = 255
        # single coord format is: start_point.x, start_point.y, end_point.x, end_point.y
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                frame = obj["frame"]
                if frame != int(img_id):
                    continue
                else:

                    left_coords = obj["yellow_lines"]
                    right_coords = obj["blue_lines"]
                    # plt.imshow(img_arr)
                    #[0] because we have [[[]]]
                    for left_segment in left_coords:
                        line_start = (int(left_segment[0][0]), int(left_segment[0][1]))
                        line_end = (int(left_segment[1][0]), int(left_segment[1][1]))
                        # if left_segment[3] > left_segment[1]:
                        #     # print("detected")
                        #     continue
                        # print(left_segment[1], left_segment[3])
                        cv2.line(left_mask, line_start, line_end, color, thickness)

                    left_mask = left_mask[300:720, :]

                    # left_mask = cv2.resize(left_mask, (448, 448))
                    # left lane is already drawn
                    blue_lane = np.argmax(left_mask, axis=1)

                    right_mask = np.zeros((height, width), dtype=np.uint8)

                        # cv2.line(mask, line2_start, line2_end, color, thickness)
                        # plt.plot([left_segment[0], left_segment[2]], [left_segment[1], left_segment[3]], color='b')
                    for right_segment in right_coords:
                        line_start = (int(right_segment[0][0]), int(right_segment[0][1]))
                        line_end = (int(right_segment[1][0]), int(right_segment[1][1]))
                        # if right_segment[3] > right_segment[1]:
                        #     # print("detected")
                        #     continue
                        cv2.line(right_mask, line_start, line_end, color, thickness)

                    right_mask = right_mask[300:720, :]

                    # right_mask = cv2.resize(right_mask, (448, 448))
                    yellow_lane = np.argmax(right_mask, axis=1)
                    # for sure break and for time also
                    break
            # print(f"image name is: {img_name}")
        # TODO 419
        blue_points = blue_lane[rows]

        yellow_points = yellow_lane[rows]

        blue_lane[~rows] = 0
        yellow_lane[~rows] = 0
        mask = {}
        mask["left_lane"] = left_mask #[rows]
        mask["right_lane"] = right_mask #[rows]
        # points = np.concatenate((blue_points, yellow_points), axis=0)
        return mask

def gaussian_k(x0,sigma, width):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float) ## (width,)
        # print(f"width is {width}")
        return np.exp(-(x - x0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def create_heatmap(mask, img_name):
    
    # heatmap = gaussian_k(3, 5, 15)
    # print(heatmap)
    # cv2.imwrite('heatmap.png', heatmap)
    
    # mask_array = cv2.imread('/home/capurjos/big_dataset_w_fsoco/masks/acceleration_czech_2?110_0.png', cv2.IMREAD_GRAYSCALE)
    # _, contours, _ = cv2.findContours(mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(mask_array, contours, -1, (255, 0, 0), 2)
    # cnt = contours[0]
    # epsilon = 0.001 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # x = approx[:, 0, 0]
    # y = approx[:, 0, 1]

    # # Plot vertices as scatter plot
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal', adjustable='box')
    # fig, ax = plt.subplots()
    # # # plt.show()
    # ax.scatter(x, y)
    # print(mask_array)
    # print(f"Mask array contains at least one '1': {np.any(mask_array == 1)}")
    # mask_array = cv2.GaussianBlur(mask_array, (201, 201), 2)
    # print(mask_array)
    # ax.imshow(mask_array, cmap='gray')
    # plt.savefig('heatmap.png')
    
# Apply the color map to the binary mask image

    # heatmap = cv2.applyColorMap(cv_left_lane_mask, cv2.COLORMAP_JET)
    # print(mask)
    left_lane_mask = mask["left_lane"]
    ll = left_lane_mask
    # print(left_lane_mask.shape)

    # print(f"left heatmap shape is {left_heatmap.shape}")
    # left_heatmap = cv2.applyColorMap(left_heatmap, cv2.COLORMAP_JET)
    # Save the heatmap image
    # cv2.imwrite('left_heatmap.png', left_heatmap)
    # print(left_lane_mask.shape)
    right_lane_mask = mask["right_lane"]
    ##################
    # left_lane = np.empty((0, left_lane_mask.shape[1]))
    # for row in left_lane_mask[rows]:
    #     if 255 in row:
    #         # TODO [0] for synthetic remove
    #         idx = np.where(row == 255)[0][0]
    #         # print(f"idx is {idx}")
    #         # print(f"shape of row is {row.shape}")
    #         row_heatmap = gaussian_k(idx, 20, left_lane_mask.shape[1])
    #         # print(f"sum is {np.sum(row_heatmap)}")
    #         if left_lane.shape[0] == 0:
    #             left_lane = row_heatmap
    #         else:
    #             left_lane = np.vstack((left_lane, row_heatmap))
    #     else:
    #         # print("not in")
    #         if left_lane.shape[0] == 0:
    #             left_lane = row
    #         else:
    #             left_lane = np.vstack((left_lane, row))

    # right_lane = np.empty((0, right_lane_mask.shape[1]))
    # for row in right_lane_mask[rows]:
    #     if 255 in row:
    #         idx = np.where(row == 255)[0][0]
    #         row_heatmap = gaussian_k(idx, 20, right_lane_mask.shape[1])
    #         if right_lane.shape[0] == 0:
    #             right_lane = row_heatmap
    #         else:
    #             right_lane = np.vstack((right_lane, row_heatmap))
    #     else:
    #         if right_lane.shape[0] == 0:
    #             right_lane = row
    #         else:
    #             right_lane = np.vstack((right_lane, row))
            # right_lane.append(row)
    ##################
    # print(f"shape of left lane is {right_lane.shape}")
    # left_lane = np.array(left_lane)
    # right_lane = np.array(right_lane)
    kernel_size = 5
    sigma = 180
    kernel = cv2.getGaussianKernel(kernel_size, sigma).transpose()
    # kernel /= np.sum(kernel)


    # Apply the kernel to the image
    # mask = np.isin(np.arange(left_lane_mask.shape[0]), rows)
    # left_lane_mask = np.where(mask[:, np.newaxis], left_lane_mask, 0)
    left_heatmap = cv2.filter2D(left_lane_mask, -1, kernel)

    mask = np.isin(np.arange(right_lane_mask.shape[0]), rows)
    right_lane_mask = np.where(mask[:, np.newaxis], right_lane_mask, 0)
    right_heatmap = cv2.filter2D(right_lane_mask, -1, kernel)
    ##
    plt.clf()
    # mask_array = cv2.GaussianBlur(ll, (201, 201), 2)
    right_heatmap_sc = np.interp(left_heatmap, (left_heatmap.min(), left_heatmap.max()), (0, 1000))
    heatmap = plt.imshow(right_heatmap_sc, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.show()
    plt.savefig("ground_truth_heatmaps/" + img_name)
    ##
    left_heatmap = left_heatmap.reshape(420, 1280)
    right_heatmap = right_heatmap.reshape(420, 1280)
    # cv2.imwrite("before.png", left_heatmap)
    left_heatmap = cv2.resize(left_heatmap, (256, 420))
    right_heatmap = cv2.resize(right_heatmap, (256, 420))
    left_heatmap = np.interp(left_heatmap, (left_heatmap.min(), left_heatmap.max()), (0, 1000))
    right_heatmap = np.interp(right_heatmap, (right_heatmap.min(), right_heatmap.max()), (0, 1000))
    left_heatmap = left_heatmap[rows]
    right_heatmap = right_heatmap[rows]


    # cv2.imwrite("before.png", left_heatmap)
    # left_heatmap = cv2.resize(left_lane, (256, 20))
    # right_heatmap = cv2.resize(right_lane, (256, 20))
    # print(np.sum(left_heatmap, axis=1))
    # print(left_heatmap.shape)
    
    # cmap = cv2.COLORMAP_JET  # replace with your desired colormap
    # heatmap = cv2.applyColorMap(data_norm, cmap)
    # print(f"max value in heatmap is {np.max(data_norm)}")
    # cv2.imwrite('aaa.png', heatmap)
    # heatmap_norm = cv2.normalize(left_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # cv2.imwrite('left_heatmap_colored.png', heatmap_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv_left_lane_mask = cv2.cvtColor(right_lane_mask, cv2.COLOR_GRAY2BGR)
    # right_heatmap = cv2.GaussianBlur(right_lane_mask, (191, 11), 11)


    # right_heatmap = cv2.applyColorMap(right_heatmap, cv2.COLORMAP_JET)
    # cv2.imwrite('right_heatmap.png', right_heatmap)
    # print(f"left heatmap shape is {left_heatmap.shape}")
    # print(f"right heatmap shape is {right_heatmap.shape}")
    heatmap = {"left_lane_heatmap": left_heatmap, "right_lane_heatmap": right_heatmap}
    return heatmap

if __name__ == "__main__":
    path_to_json = os.path.join("/home/capurjos/modified_labels/json_masks", "autox_try_1_czech430_0.json")
    mask = get_mask(path_to_json)
    create_heatmap(mask)
    # get_heatmap_points_from_label()