import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from dataset import fsDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import gc
import argparse
import cv2
import os
import datetime
from unet_model import UNet
import wandb
from torchinfo import summary

# import torchvision.models as models




def print_log_info(dataset, trn_size, val_size):
    print('Dataset size is {0}'.format(len(dataset)))
    print('Training size is {0}'.format(trn_size))
    print('Validation size is {0}'.format(val_size))



def plot_imgs(image, pred_left_lane, pred_right_lane, true_left_lane, true_right_lane, rows, indices, epoch):
    plt.clf()
    plt.plot(pred_left_lane, rows, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
    plt.plot(pred_right_lane, rows,  marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
    plt.plot(true_left_lane, rows, "b", true_right_lane, rows, "y")
    # plt.plot(pred_left_lane[left_indices], rows[left_indices], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
    # plt.plot(pred_right_lane[right_indices], rows[right_indices],  marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
    # plt.plot(true_left_lane[left_indices], rows[left_indices], "b", true_right_lane[right_indices], rows[right_indices], "y")

    plt.imshow((image.cpu().permute(1, 2, 0))) #.numpy().astype(np.uint8)))

    plt.savefig("predicted_imgs/epoch_" + str(epoch) + "_" + str(random.randint(0, 900)) + ".png")





def train(epochs: int, batch_sz: int, learning_rate:int, synthetic: bool,
          pretrained: bool, dataset_length: int, directory: str, img_scale: float, penalize_undefined_parts: bool, experiment):

    torch.cuda.empty_cache()
    gc.collect()

    # dir_img = Path('/home/capurjos/Pytorch-UNet/cropped_imgs_raw/imgs')
    # dir_mask = Path('/home/capurjos/Pytorch-UNet/cropped_imgs_raw/masks')
    if synthetic:
        # dir_img = Path('/home/capurjos/synt_data/imgs')
        # dir_mask = Path('/home/capurjos/synt_data/labels')
        dir_img = Path('/home/capurjos/synthetic_dataset/minimal_synthetic/imgs')
        dir_mask = Path('/home/capurjos/synthetic_dataset/minimal_synthetic/labels')
    else:
        dir_img = Path('/home/capurjos/modified_labels/imgs')
        dir_mask = Path('/home/capurjos/modified_labels/json_masks')

    # os.mkdir("losses/" + directory)
    os.mkdir("predicted_imgs/" + directory)
    # os.mkdir("smooth_grad_heatmaps/" + directory)

    num_segments_per_line = 20
    dataset = fsDataset(dir_img, dir_mask, synthetic)

    trn_size = int(0.9 * dataset_length)
    # trn_size = 5000
    val_size = int(0.1 * dataset_length)

    print_log_info(dataset, trn_size, val_size)
    train_losses = []
    val_losses = []

    trn_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [trn_size, val_size, len(dataset) - trn_size - val_size])
    # val_dataset = trn_dataset
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_sz,
                                             shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_sz,
                                             shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device {0}".format(device))
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18').to(device)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5)).to(device)
    # model = UNet(3, 1).to(device)

    # model = ResNet50NoAvgPool().to(device)
    # model.features = nn.Sequential(*list(model.features._modules.values())[:-1])
    # # model = ResNet18_noavgpool().to(device)
    # # model = ModifiedUNet


    num_ftrs = model.fc.in_features
    width = 320
    height = 419
    num_channels = 3
    # print(f"input number of neurons in last layer is {25088, 2 * num_segments_per_line * width}")

    model.fc = nn.Linear(12800 , 2 * num_segments_per_line * width).to(device)
    summary(model, input_size=(batch_sz, num_channels, 420, 1280))
    if penalize_undefined_parts:
        print("Penalizing all parts in the image..")
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()
    # criterion = RMSLELoss()
    rows = np.linspace(height, 0, num_segments_per_line).astype(int)
    global_step = 0
    for epoch in range(1, epochs + 1):
        print("Epoch {0}".format(epoch))
        model.train()
        train_loss = 0.0
        with tqdm(total=trn_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i_batch, batch in enumerate(trn_loader):
                # print(f"Epoch {epoch}. Batch: {i_batch}")
                images = batch["image"]
                filenames = batch["filename"]
                # points = batch["left_lane_heatmap"].float()
                # create 1D gaussian for each point, so each point of lane is represented as 1x1280
                left_lane_heatmap = batch["left_lane_heatmap"].float()
                # print(f"left lane gt heatmap shape is {left_lane_heatmap.shape}")
                right_lane_heatmap = batch["right_lane_heatmap"].float()

                # print(f" heatmaps are the same: {left_lane_heatmap.argmax(axis=2).unique() == right_lane_heatmap.argmax(axis=2).unique()}")
                images, left_lane_heatmap, right_lane_heatmap = images.to(device), left_lane_heatmap.to(device), right_lane_heatmap.to(device)
                net_output = model(images).float()
                if penalize_undefined_parts:
                    try:
                        # print("loss for rw dataset")
                        left_lane_nn_output, right_lane_nn_output = np.split(net_output, 2, axis=1)
                        left_lane_nn_output = left_lane_nn_output.reshape(batch_sz, 20, width)
                        right_lane_nn_output = right_lane_nn_output.reshape(batch_sz, 20, width)
                        loss = criterion(left_lane_nn_output.flatten(), left_lane_heatmap.flatten())
                        loss += criterion(right_lane_nn_output.flatten(), right_lane_heatmap.flatten())
                        optimizer.zero_grad()
                        loss.backward()

                        optimizer.step()
                        train_loss += loss.item()
                        pbar.update(images.shape[0])
                        global_step += 1
                        experiment.log({
                                'train loss': loss.item(),
                                'step': global_step,
                                'epoch': epoch
                            })
                        # TODO do i need to explicitly tell that this part of code is from PyTorch UNET?
                        pbar.set_postfix(**{'loss (batch)': loss.item()})
                        torch.cuda.empty_cache()
                        gc.collect()
                    except RuntimeError:
                        continue
        
                else:
                    try:
                        left_indices = torch.where(left_lane_heatmap.argmax(axis=2) > 2)
                        right_indices = torch.where((right_lane_heatmap.argmax(axis=2) < 1277) & (right_lane_heatmap.argmax(axis=2) > 2))
                        left_lane_nn_output, right_lane_nn_output = np.split(net_output, 2, axis=1)
                        left_lane_nn_output = left_lane_nn_output.reshape(batch_sz, 20, width)
                        right_lane_nn_output = right_lane_nn_output.reshape(batch_sz, 20, width)
                        # print(f"left lane nn output is {left_lane_nn_output.shape}")
                        # print(f"left heatmap outpit shape is {left_lane_heatmap.shape}")
                        # print(left_lane_nn_output[left_indices].shape)
                        # print(f"shape of nn left output is: {left_lane_nn_output.flatten().shape}")
                        # print(f"shape of left lane heatmap is {left_lane_heatmap.flatten().shape}")
                        loss = criterion(left_lane_nn_output[left_indices].flatten(), left_lane_heatmap[left_indices].flatten())
                        loss += criterion(right_lane_nn_output[right_indices].flatten(), right_lane_heatmap[right_indices].flatten())
                        optimizer.zero_grad()
                        loss.backward()
                        # TODO
                        # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                        optimizer.step()
                        train_loss += loss.item()
                        torch.cuda.empty_cache()
                        gc.collect()
                    except RuntimeError:
                        continue



                # plt.savefig("predicted_imgs/epoch_" + str(epoch) + "_" + str(random.randint(0, 900)) + ".png")
                # loss = criterion(net_output[indices], points[indices])
                # left_lane_nn_output.shape is width * 30
                # print(net_output.shape)
                # TODO check if this works correctly for batch > 1



                if i_batch % 10 == 0:
                    print('[TRN] Train epoch: {}, batch: {}\tLoss: {:.4f}'.format(
                        epoch, i_batch, loss.item()))
            print(f"Training loss per epoch is {train_loss / len(trn_loader)}")
            train_losses.append(train_loss / len(trn_loader))

            model.eval()

            val_loss = 0.0
            # TODO again penalizing parts etc.
            with torch.no_grad():
                # if epoch < 25:
                #     continue
                # TODO switch to val_loader
                for i_batch, batch in enumerate(val_loader):
                    images = batch["image"]
                    left_lane_heatmap = batch["left_lane_heatmap"].float()
                    right_lane_heatmap = batch["right_lane_heatmap"].float()
                    # print(f"left lane heatmap is {left_lane_heatmap.unique()}")
                    # print(f"right lane heatmap is {right_lane_heatmap.unique()}")
                    filenames = batch["filename"][0]

                    images, left_lane_heatmap, right_lane_heatmap = images.to(device), left_lane_heatmap.to(device), right_lane_heatmap.to(device)
                    net_output = model(images).float()
                    left_lane_nn_output, right_lane_nn_output = np.split(net_output, 2, axis=1)
                    ##########
                    # : tuple(torch.tensor, torch.tensor)
                    left_indices = torch.where(left_lane_heatmap.argmax(axis=2) > 2)
                    right_indices = torch.where((right_lane_heatmap.argmax(axis=2) < width - 1) & (right_lane_heatmap.argmax(axis=2) > 2))
                    # print(f"net output shape is {net_output.shape}")
                    left_lane_nn_output, right_lane_nn_output = np.split(net_output, 2, axis=1)
                    try:
                        left_lane_nn_output = left_lane_nn_output.reshape(batch_sz, 20, width)
                        right_lane_nn_output = right_lane_nn_output.reshape(batch_sz, 20, width)
                    except RuntimeError:
                        continue

                    ###########

                    if penalize_undefined_parts:
                        loss = criterion(left_lane_nn_output.flatten(), left_lane_heatmap.flatten())
                        loss += criterion(right_lane_nn_output.flatten(), right_lane_heatmap.flatten())
                    else:
                        loss = criterion(left_lane_nn_output[left_indices].flatten(), left_lane_heatmap[left_indices].flatten())
                        loss += criterion(right_lane_nn_output[right_indices].flatten(), right_lane_heatmap[right_indices].flatten())
                    experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'step': global_step,
                                    'epoch': epoch,
                                    'validation loss': loss.item()
                                })
                    # indices = torch.where(points > 2)
                    # # todo...
                    # loss = criterion(net_output[indices], points[indices])

                    # if i_batch % 3 > 0:
                        # print(f"length of images is {len(images)}")
                    # left_indices = left_indices.cpu()
                    # right_indices = right_indices.cpu()
                    left_lane_heatmap = left_lane_heatmap.cpu()
                    right_lane_heatmap = right_lane_heatmap.cpu()
                    # print(f"shape of left lane heatmap is {left_lane_heatmap.shape}")
                    # print(f"shape of left indices is {left_indices}")
                    for i in range(len(images)):
                        prediction = net_output[i].cpu()
                        # TODO why [1] index?
                        # cur_left_indices = left_indices[1][np.where(left_indices[0].cpu() == i)].cpu()

                        pred_left_lane, pred_right_lane = np.split(prediction, 2)
                        pred_left_lane = pred_left_lane.reshape(20, width)
                        pred_right_lane = pred_right_lane.reshape(20, width)
                        left_lane_pts = np.argmax(pred_left_lane, axis=1)
                        right_lane_pts = np.argmax(pred_right_lane, axis=1)
                        # left_heatmap_mask = np.zeros((256, 420))
                        # # TODO probably wont work
                        # left_heatmap_mask[[rows], :] = pred_left_lane
                        # right_heatmap_mask = np.zeros((256, 420))
                        # right_heatmap_mask[[rows], :] = pred_right_lane
                        # # print(filenames)
                        # # TODO maybe use cv2.resize
                        # left_heatmap_mask = left_heatmap_mask.resize(420, 1280)
                        # right_heatmap_mask = right_heatmap_mask.resize(420, 1280)
                        filename = filenames[i].split("/")[-1]
                        # if epoch > 10:
                        #     plt.clf()
                        #     heatmap = plt.imshow(left_heatmap_mask, cmap='hot', interpolation='nearest')
                        #     plt.colorbar(heatmap)
                        #     # plt.show()
                        #     plt.savefig("heatmaps/" + directory + "/left_heatmap_epoch_" + str(epoch) + "_" + filename)
                        #     # ---
                        #     plt.clf()
                        #     heatmap = plt.imshow(right_heatmap_mask, cmap='hot', interpolation='nearest')
                        #     plt.colorbar(heatmap)
                        #     # plt.show()

                        #     plt.savefig("heatmaps/" + directory + "/right_heatmap_epoch_" + str(epoch) + "_" + filename)

                        if epoch > 30:
                            scaling_param = 1280 / width
                            plt.clf()
                            plt.plot(left_lane_pts * scaling_param, rows, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                            plt.plot(right_lane_pts * scaling_param, rows,  marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                            plt.plot(left_lane_heatmap[i].argmax(axis=1) * scaling_param, rows, "b")
                            plt.plot(right_lane_heatmap[i].argmax(axis=1) * scaling_param, rows, "y")
                            plt.imshow((images[i].cpu().permute(1, 2, 0))) #.numpy().astype(np.uint8)))
                            plt.savefig("predicted_imgs/" + directory + "/epoch_" + str(epoch) + "_" + filename)

                    val_loss += loss.item()
                try:
                    print(f"Validation loss per epoch is {val_loss / len(val_loader)}")
                    val_losses.append(val_loss / len(val_loader))
                except ZeroDivisionError:
                    print(f"Length of validation dataset is zero.")
        # TODO synthetic and real_world should have this directory in losses etc.
        if synthetic:
            path_to_model = "models/synthetic/" + directory
        else:
            path_to_model = "models/real_world/" + directory

        try:
            if val_losses[-1] == min(val_losses):
                torch.save(model.state_dict(), path_to_model + "_model_best_epoch_" + str(epoch) + ".pt")
        except:
            pass
        torch.save(model.state_dict(), path_to_model + "_model_epoch_" + str(epoch) + ".pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        # SOME of these arguments are taken from Milesial U-Net code
        # I inspired from this code since I liked the structure, when I trained the UNet
    parser.add_argument('--synthetic', action='store_true',
                        help='Train with synthetic data')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use prretrained model trained on synthetic dataset')
    parser.add_argument('--debug', action='store_true',
                        help='True if you want to print additional information')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3, help='Learning rate', dest='lr')
    parser.add_argument('--penalize',  action='store_true', help='Penalize borders of the road that are ambiguous')
    args = parser.parse_args()
    experiment = wandb.init(project='ResNet points', resume='allow', anonymous='must')
    timestamp = datetime.datetime.now()
    # dataset_length = 2000
    if args.synthetic:
    # dataset_length = 1000
    # train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length))
    # dataset_length = 5000
    # train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length))
    # dataset_length = 10000
    # train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length))
        dataset_length = 1000
        train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length) + "_" + str(timestamp))
    else:
        dataset_length = 850
        directory = "real_world_" + str(dataset_length) + "_" + str(timestamp)
        if args.pretrained:
            directory = "real_world_" + "pretrained_" + str(dataset_length) + "_" + str(timestamp)
        # TODO heatmaps currently not support penalizing undefined parts and scaling images.. change!!
        train(args.epochs, args.batch_size, args.lr, args.synthetic,
          args.pretrained, dataset_length, directory, args.scale, args.penalize, experiment)
        # train(args.synthetic, args.pretrained, dataset_length, directory)
    # train(args.synthetic, args.pretrained, dataset_length=dataset_length, directory=)
















# TODO shape of gaussian?????? - bigger the better for convergence?
# TODO visualise gaussians
# TODO reshape images ... 224x224 - maybe not necessary
# TODO - predictions on test dataset!!!
# TODO prediction labeling tool
# TODO sample more datat to UNEt
# TODO modify unet and interpolate beginning
# TODO pretraining on synthetized dataset, sample more data to synthetized dataset
# TODO !!!!!! synthetized dataset - solve for points where track is not defined - interpolate !!!!!!
# TODO evaluate everything
# TODO cleanup code
# TODO create not equal size indices..
# TODO correct val dataset that doesnt contain anything else
# TODO plot heatmaps!!!!
# TODO visualise kernels and decision parts in image
# solve heatmaps and solve resnet so it can be used for real.. i.e. curvatures
# 1. synthetic dataset - prepare more samples and remove bad labels
# 2. prediction labeling tool
# 3. predictions on test dataset
# 4. prediction on synthetized dataset - does it work better than for real dataset? yes, but large dataset is needed
# 4. later - sample for unet??
# 5. later - why heatmaps work so poorly?
# 6. gaussian should have same value for every point