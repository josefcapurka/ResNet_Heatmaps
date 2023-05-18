import logging
import numpy as np
import torch
from PIL import Image
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision import transforms
import logging
import json
import cv2
from heatmap import create_heatmap, get_mask, get_synt_mask_measurements


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename).convert("RGB")


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')




class fsDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, is_synthetic: bool):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.is_synthetic = is_synthetic
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        # self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        self.transforms = transforms.Compose([transforms.ToTensor()])

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # self.detector = Detector()
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)


    @staticmethod
    def get_mask_measurements(label_path, img_name):
        height = 420
        width = 1280
        left_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the lines on the mask with white color (255)
        thickness = 2
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
            
            # left_mask = cv2.resize(left_mask, (224, 224))
            # left lane is already drawn
            blue_lane = np.argmax(left_mask, axis=1)
            
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
            
            # right_mask = cv2.resize(right_mask, (224, 224))
            yellow_lane = np.argmax(right_mask, axis=1)
            # print(f"image name is: {img_name}")
            # cv2.imwrite("visualise_labels/" + str(img_name) + ".png", left_mask + right_mask)
        # todo 419?
        rows = np.linspace(419, 50, 20).astype(int)
        blue_points = blue_lane[rows]

        yellow_points = yellow_lane[rows]
        points = np.concatenate((blue_points, yellow_points), axis=0)
        return points
    
    # def get_points(self, path_to_img, path_to_color_img):
    #     # print(f"path is {path_to_img[0]}")
    #     mask_array = cv2.imread(str(path_to_img[0]), cv2.IMREAD_GRAYSCALE)
    #     img = cv2.imread(str(path_to_color_img), cv2.COLOR_BGR2RGB)

    #     # Convert binary mask image to numpy array
    #     # mask_array = mask_image.numpy()

    #     # Find contours in mask image
    #     # contours, _, _ = cv2.findContours(mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contours, _ = cv2.findContours(
    #         mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #     cv2.drawContours(mask_array, contours, -1, (255, 0, 0), 2)
    #     cnt = contours[0]
    #     epsilon = 0.001 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     x = approx[:, 0, 0]
    #     y = approx[:, 0, 1]

    #     # Plot vertices as scatter plot
    #     plt.gca().invert_yaxis()
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     fig, ax = plt.subplots()
    #     ax.imshow(img)#, cmap='gray')
    #     # plt.show()
    #     ax.scatter(x, y)
    #     plt.savefig("visualise_labels/" + str(path_to_img[0].stem) + ".jpg")




    def visualise_labels(self, image, mask, img_name):
        rows = np.linspace(420, 0, 2).astype(int)
        left_lane, right_lane = np.split(mask, 2)
        # fig, ax = plt.subplots()
        plt.clf()
        # Iterate over pairs of consecutive points
        left_x_axis = []
        left_y_axis = []
        prev_point = None
        for i, (x_i, y_i) in enumerate(zip(rows, left_lane)):
            if prev_point is not None:
                # Compute the angle between the previous point and the current point
                # dx = x_i - prev_point[0]
                # dy = y_i - prev_point[1]
                # angle = np.arctan2(dy, dx) * 180 / np.pi

                # If the angle is greater than 80 degrees, skip this point
                if y_i <= 5:
                    continue    
                left_x_axis.append(x_i)
                left_y_axis.append(y_i)

                # Otherwise, plot a line segment connecting the previous point to this point
                # plt.plot([prev_point[0], x_i], [prev_point[1], y_i], color='blue')

            prev_point = (x_i, y_i)
        right_x_axis = []
        right_y_axis = []
        prev_point = None
        for i, (x_i, y_i) in enumerate(zip(rows, right_lane)):
            if prev_point is not None:
                # Compute the angle between the previous point and the current point
                # dx = x_i - prev_point[0]
                # dy = y_i - prev_point[1]
                # angle = np.arctan2(dy, dx) * 180 / np.pi

                # If the angle is greater than 80 degrees, skip this point
                if y_i >= 415:
                    continue
                right_x_axis.append(x_i)
                right_y_axis.append(y_i)

                # Otherwise, plot a line segment connecting the previous point to this point
                # plt.plot([prev_point[0], x_i], [prev_point[1], y_i], color='blue')

            prev_point = (x_i, y_i)
        # plt.clf()
        # plt.plot(left_lane, rows, "b", right_lane, rows, "y", linewidth='4')
        plt.plot(left_y_axis, left_x_axis, "b", right_y_axis, right_x_axis, "y", linewidth='4')
        plt.imshow(image)
        plt.savefig("visualise_labels/" + str(img_name))
        plt.close()

    # def print_bboxes(self, c_fpath):
    #     print(c_fpath)
    #     nn_bboxes = self.detector.run(Path(c_fpath))
    #     print(nn_bboxes)
    def visualise_label(self, img_path, label_path, img_name):
    # load image
        plt.clf()
        img = Image.open(img_path)
        img_arr = np.array(img)
        # load labels
        with open(label_path, 'r') as f:
            data = json.load(f)
            left_coords = data["left_coords"]
            right_coords = data["right_coords"]
            plt.imshow(img_arr)
            for left_segment in left_coords:
                plt.plot([left_segment[0], left_segment[2]], [left_segment[1], left_segment[3]], color='b')
            for right_segment in right_coords:
                plt.plot([right_segment[0], right_segment[2]], [right_segment[1], right_segment[3]], color='y')
            # plt.show()
            # plt.imshow(img_arr)
            plt.savefig("visualise_labels/" + str(img_name))
            plt.close()

    def __getitem__(self, idx):
        
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        # self.visualise_label(img_file[0], mask_file[0], name)
        # self.print_bboxes(img_file[0])

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img = load_image(img_file[0])
        # img = img.resize((1344, 448))
        # img = img.resize((224, 224))
        logging.info(f"Shape of input image as PIL is {img.size}")
        # mask = load_image(mask_file[0])
        # mask = mask.resize((448, 448))
        # self.get_points(mask_file, img_file[0])
        
        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        #----------
        img = np.asarray(img)
        logging.info(f"Shape of input image as nparray is {img.shape}")


        # In- and output are of the form Bs, Ch, H, W

        # print(img)
        # print(torch.as_tensor(img.copy()).float().contiguous())
        logging.info(f"Shape of input image after transpose as nparray is {img.shape}")
        # mask = np.asarray(mask)
        # mask = fsDataset.get_mask_measurements(mask_file[0], name)
        # for real data
        # mask = get_mask(mask_file[0])
        # print(mask_file)
        # for
        if self.is_synthetic:
            # print(f"filename is {str(img_file[0])}")
            img = img[300:719, :, :]
            # print(img.shape)
            mask = get_synt_mask_measurements(str(img_file[0]), self.mask_dir)
        else:
            mask = get_mask(mask_file[0])
        heatmap = create_heatmap(mask, str(img_file[0]).split("/")[-1])
        # print(f"shape of left heatmap: {heatmap['left_lane_heatmap'].shape}")
        # print(f"shape of right heatmap: {heatmap['right_lane_heatmap'].shape}")
        # self.visualise_labels(image, mask, name)
        img =  img.transpose((2, 0, 1))
        img = img / 255
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            # 'points': torch.as_tensor(mask.copy()).long().contiguous()
            'left_lane_heatmap': torch.as_tensor(heatmap["left_lane_heatmap"].copy()).float().contiguous(),
            'right_lane_heatmap': torch.as_tensor(heatmap["right_lane_heatmap"].copy()).float().contiguous(),
            'filename': [str(img_file[0])]
        }

