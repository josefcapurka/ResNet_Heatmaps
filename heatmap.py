
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
# import seaborn as sns
import json
import jsonlines
from pathlib import Path
import random
# from dataset import fsDataset
num_segments_per_line = 20
height = 419
rows = np.linspace(height, 0, num_segments_per_line).astype(int)


def get_mask(label_path):
        height = 420
        width = 1280
        left_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the lines on the mask with white color (255)
        thickness = 1
        color = 255
        # single coord format is: start_point.x, start_point.y, end_point.x, end_point.y
        with open(label_path, 'r') as f:
            # data loading and drawing lines
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

            # first_one_right = np.argmax(right_mask, axis=1)
            # modified_right_mask = []
            # set_zero_indices = np.setdiff1d(np.arange(420), rows)

            # yellow_lane = np.where(right_mask == 255)
            # _, right_indices = np.unique(yellow_lane[0], return_index=True)
            # r_indices = (yellow_lane[0][right_indices], yellow_lane[1][right_indices])
            # # print(f"image name is: {str(label_path).split('/')[-1]}")
            # left_mask = np.zeros((height, width), dtype=np.uint8)
            # right_mask = np.zeros((height, width), dtype=np.uint8)
            # left_mask[l_indices] = 255

            # left_mask[l_indices] = 255
            # right_mask[r_indices] = 255

            # cv2.imwrite("visualise_labels/" + str(label_path).split("/")[-1] + ".png", left_mask + right_mask)
        mask = {}
        first_one_indices = np.argmax(left_mask == 255, axis=1)

# create a mask for all elements after the first occurrence of 1 in each row
        # indices = np.arange(left_mask.shape[1]) > first_one_indices[:, np.newaxis]
        # left_mask[indices] = 0
        set_zero_indices = np.setdiff1d(np.arange(420), rows)
        left_mask[set_zero_indices] = 0
        left = np.zeros_like(left_mask)
        idx = np.argmax(left_mask, axis=1)
        left[np.arange(len(left_mask)), idx] = 255
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!
        left[:, 0] = 0
        # print(f"ones in arr is {np.sum(left==1)}")
        right_mask[set_zero_indices] = 0
        right = np.zeros_like(right_mask)
        idx = np.argmax(right_mask, axis=1)
        right[np.arange(len(right_mask)), idx] = 255
        right[:, 0] = 0
        # print(rows)
        mask["left_lane"] = left #[rows]
        mask["right_lane"] = right #[rows]

        return mask

def get_synt_mask_measurements(img_name, mask_dir):
        img_name = img_name.split("/")[-1]
        # example of img_name: run_261_000008
        # print(img_name)
        run_number = img_name.split("_")[1]
        # print(f"run number is {run_number}")
        img_id = img_name.split("_")[2][0:-4]
        dir_mask = Path(mask_dir)
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
                    # blue_lane = np.argmax(left_mask, axis=1)

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
                    # yellow_lane = np.argmax(right_mask, axis=1)
                    # for sure break and for time also
                    break
            # print(f"image name is: {img_name}")
        # TODO 419
        mask = {}
        set_zero_indices = np.setdiff1d(np.arange(420), rows)
        left_mask[set_zero_indices] = 0
        left = np.zeros_like(left_mask)
        idx = np.argmax(left_mask, axis=1)
        left[np.arange(len(left_mask)), idx] = 255
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!
        left[:, 0] = 0
        # print(f"ones in arr is {np.sum(left==1)}")
        right_mask[set_zero_indices] = 0
        right = np.zeros_like(right_mask)
        idx = np.argmax(right_mask, axis=1)
        right[np.arange(len(right_mask)), idx] = 255
        right[:, 0] = 0
        # print(rows)
        mask["left_lane"] = left #[rows]
        mask["right_lane"] = right #[rows]
        return mask



def create_heatmap(mask, img_name):
    left_lane_mask = mask["left_lane"]
    left_heatmap = cv2.GaussianBlur(left_lane_mask, (101, 101), 2)


    right_lane_mask = mask["right_lane"]
    right_heatmap = cv2.GaussianBlur(right_lane_mask, (101, 101), 2)
    # plt.clf()
    # heatmap = plt.imshow(right_heatmap, cmap='hot', interpolation='nearest')
    # plt.colorbar(heatmap)
    # # # plt.show()
    # plt.savefig("heatmaps_labels/right" + img_name)

    # bag of tricks
    height, width = (420, 320)
    # TODO visualise heatmaps after this..
    left_heatmap = cv2.resize(left_heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    right_heatmap = cv2.resize(right_heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    # Scaling
    left_heatmap = np.interp(left_heatmap, (left_heatmap.min(), left_heatmap.max()), (0, 1000))
    right_heatmap = np.interp(right_heatmap, (right_heatmap.min(), right_heatmap.max()), (0, 1000))
    # print(f"left heatmap shape is {left_heatmap.shape}")
    left_heatmap = left_heatmap[rows]

    right_heatmap = right_heatmap[rows]
    


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
    # print(f"values are unique: {leaft_heatmap.unique()}")
    heatmap = {"left_lane_heatmap": left_heatmap, "right_lane_heatmap": right_heatmap}
    return heatmap

if __name__ == "__main__":
    path_to_json = os.path.join("/home/josef/Documents/eforce/test_dtst/", "autox_try_1_czech430_0.json")
    mask = get_mask(path_to_json)
    create_heatmap(mask, "teest.png")
    # get_heatmap_points_from_label()