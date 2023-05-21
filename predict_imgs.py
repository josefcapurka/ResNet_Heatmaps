import argparse
import logging
import os
import torch.nn as nn

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


from visualise_nn_output import visualise
# Create the model




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/home/capurjos/ResNet-CNN/model.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input_dir', '-i', metavar='INPUT', default='/home/capurjos/Pytorch-UNet/FSG', help='Directory of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    
    return parser.parse_args()




def predict_imgs():
    args = get_args()
    num_segments_per_line = 30
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50') #.to(device)

    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2 * num_segments_per_line) #.to(device)
    # model = CustomResnet()
    # print(model)
    # model_dict_1 = model.state_dict()
    # print(model_dict)
    print("________________")
    rows = np.linspace(420, 50, num_segments_per_line)
    # Load state_dict
    model.load_state_dict(torch.load(args.model), strict=True)
    # model_dict_2 = model.state_dict()
    # print(model_dict_1model_dict_2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
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
    print(type(args.input_dir))
    """
    1. output from nn is 
    """
    with torch.no_grad():
    
        for filename in os.listdir(args.input_dir):
            if ".png" in filename:
                print("d")
                img = Image.open(os.path.join(args.input_dir, filename))
                # print(img.shape)
                img = img.convert('RGB') #.to(device)
                # img = img.resize((448, 448))
                
                # unsqueeze batch dimension, in case you are dealing with a single image
                # # print(trans.shape)
                input = trans(img)
                input = input.unsqueeze_(0)
                input = input.to(device) # "cuda:0")
                # input = input
                # input = input.to(device)
                output = model(input)
                print(output)
                output_filename = "/home/capurjos/ResNet-CNN/output_dir/" + filename
                visualise(img, output.cpu(), rows, output_filename)

if __name__ == "__main__":
    predict_imgs()