import argparse
import logging
import os
import torch.nn as nn

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


from visualise_nn_output import visualise
from unet_model import UNet
import cv2
# Create the model




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/home/capurjos/ResNet-CNN/model.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input_dir', '-i', metavar='INPUT', default='/home/capurjos/Pytorch-UNet/FSG', help='Directory of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument("--evaluate_accuracy", '-e', action='store_true')
    return parser.parse_args()

def iou_numpy(outputs, labels):
    intersection = np.logical_and(outputs, labels)
    union = np.logical_or(outputs, labels)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def draw_polygon_from_pts(polygon_pts, save_path):
    # polygon_pts = polygon_pts.astype(int)


# Create a new image of size 300x300 with a white background
    # img = Image.new('RGB', (1280, 420), color='white')
    # print(polygon_pts)
    # Draw the polygon on the image
    img = np.zeros((420, 1280), dtype=np.uint8)
    # print(f"shape is {polygon_pts.shape}")
    copied_img = img.copy()
    cv2.fillPoly(copied_img, pts=np.int32([polygon_pts]), color=(255, 255, 255))
    cv2.imwrite(save_path, copied_img)
    return copied_img


def predict_imgs():
    args = get_args()
    num_segments_per_line = 20
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50') #.to(device)
    # model = UNet(3, 1)
    # num_ftrs = model.fc.in_features
    width = 320
    height = 419
    device = torch.device('cuda')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18').to(device)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5)).to(device)
    model.fc = nn.Linear(12800 , 2 * num_segments_per_line * width).to(device)
    print(f"Using model {args.model}")
    # model = CustomResnet()
    # print(model)
    # model_dict_1 = model.state_dict()
    # print(model_dict)
    print("________________")
    # rows = np.linspace(420, 50, num_segments_per_line)
    rows = np.linspace(height, 0, num_segments_per_line).astype(int)
    # Load state_dict
    model.load_state_dict(torch.load(args.model), strict=True)
    # model_dict_2 = model.state_dict()
    # print(model_dict_1model_dict_2)
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    print(f'Using device {device}')
    print(model)
    iou = 0
    # model = ResNet(3, 18, BasicBlock, 60).to(device)
    model.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    trans = transforms.Compose([
            # other transforms
            transforms.ToTensor(),
            # lambda x: x*255
        ])
    print(f"input directory is {args.input_dir}")
    print(f"output directory is {args.output}")
    # size = (int(img_scale * 1280), int(img_scale * 420))
    masks_dir = os.path.join(os.path.join(args.input_dir, '..'), "masks")
    imgs_processed = 0
    with torch.no_grad():

        for filename in os.listdir(args.input_dir):

            if ".png" in filename:

                img = Image.open(os.path.join(args.input_dir, filename))
                if img.size[0] == 1280 and img.size[1] == 720:
                    img = img.crop((0, 300, 1280, 720))
                # TODO image.bicubic??
                # img = img.resize(size)
                # print(img.shape)
                img = img.convert('RGB') #.to(device)
                img = np.asarray(img)
                # img = img[299:719, :, :]
                # img = img.resize((448, 448))

                # unsqueeze batch dimension, in case you are dealing with a single image
                # # print(trans.shape)
                # input = trans(img)
                # img =  img.transpose((2, 0, 1)) # self.preprocess(img, self.scale, is_mask=False)
                # img = img / 255
                input = trans(img).float()
                input = input.unsqueeze(0)
                # print(input.shape)
                input = input.to(device) # "cuda:0")
                # input = input
                # input = input.to(device)

                output = model(input).cpu().numpy()
                imgs_processed += 1
                print(f"shape of output is {output.shape}")
                pred_left_lane, pred_right_lane = np.split(output[0], 2)
                pred_left_lane = pred_left_lane.reshape(20, width)
                pred_right_lane = pred_right_lane.reshape(20, width)
                left_lane_pts = np.argmax(pred_left_lane, axis=1)
                right_lane_pts = np.argmax(pred_right_lane, axis=1)
                scaling_param = 1280 / width
                scaled_left_lane_pts = left_lane_pts * scaling_param
                scaled_right_lane_pts = right_lane_pts * scaling_param

                output_filename = str(args.output) + filename


                visualise(img, scaled_left_lane_pts, scaled_right_lane_pts, rows, output_filename)
    print(f"IoU is {iou / imgs_processed}")

if __name__ == "__main__":
    predict_imgs()