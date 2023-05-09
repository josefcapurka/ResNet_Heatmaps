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


def plot_losses(epoch, train_losses, val_losses):
    plt.clf()
    plt.plot(range(1,epoch + 1),train_losses,'b-',label='train_loss')
    plt.plot(range(1,epoch + 1),val_losses,'g-',label='val_loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig("losses/" + str(epoch) + "_" + "losses" + ".png")

class ResNet18_noavgpool(torch.nn.Module):
    def __init__(self):
        super(ResNet18_noavgpool, self).__init__()
        # TODO resnet 50
        # resnet18 = models.resnet50(pretrained=True)
        self.resnet18 = models.resnet18(pretrained=True)
        self.features = torch.nn.Sequential(*list(self.resnet18.children())[:-2])
        width = 1280
        num_segments_per_line = 2
        print(2 * num_segments_per_line * width)
        self.l1 = torch.nn.Linear(2048, 2 * num_segments_per_line * width)
        # self.l2 = torch.nn.Linear(90000, 60)
        # self.classifier = torch.nn.Linear(1146880 , 60)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        # x = self.l2(x)
        return x


class ResNet50NoAvgPool(nn.Module):
    def __init__(self):
        super(ResNet50NoAvgPool, self).__init__()
        # Load the original ResNet50 model
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        # Remove the last layer (avgpool + fc) from the original ResNet50 model
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        width = 1280
        num_segments_per_line = 5
        print(2 * num_segments_per_line * width)
        self.l1 = torch.nn.Linear(286720, 2 * num_segments_per_line * width)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        return x

class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, y_hat, y):
        """
        Compute the mean squared error between predicted and ground truth heatmaps.

        Args:
            y_hat: predicted heatmap (batch_size, num_landmarks, height, width)
            y: ground truth heatmap (batch_size, num_landmarks, height, width)

        Returns:
            loss: mean squared error between predicted and ground truth heatmaps (scalar)
        """
        mse_loss = nn.MSELoss()
        # mae_loss = nn.L1Loss()
        # loss = mae_loss(y_hat, y)
        loss = mse_loss(y_hat, y)
        return loss

    
def train(synthetic: bool, pretrained: bool, dataset_length, directory):
    # torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    gc.collect()

    batch_sz = 8

    learning_rate = 0.01
    epochs = 100
    # dir_img = Path('/home/capurjos/Pytorch-UNet/cropped_imgs_raw/imgs')
    # dir_mask = Path('/home/capurjos/Pytorch-UNet/cropped_imgs_raw/masks')
    if synthetic:
        dir_img = Path('/home/capurjos/synt_data/imgs')
        dir_mask = Path('/home/capurjos/synt_data/labels')
    else:
        dir_img = Path('/home/capurjos/modified_labels/imgs')
        dir_mask = Path('/home/capurjos/modified_labels/json_masks')
    
    os.mkdir("losses/" + directory)
    os.mkdir("predicted_imgs/" + directory)
    # os.mkdir("smooth_grad_heatmaps/" + directory)
    img_scale = 1
    num_segments_per_line = 20
    dataset = fsDataset(dir_img, dir_mask, synthetic)

    trn_size = int(0.9 * dataset_length)
    # trn_size = 5000
    val_size = int(0.1 * dataset_length)
    # trn_size = 2000 # because 880 is divisible by batch size 
    # TODO solve this error File "train.py", line 211, in train
#     left_lane_nn_output = left_lane_nn_output.reshape(batch_sz, 15, width)
# RuntimeError: shape '[8, 15, 1280]' is invalid for input of size 19200

    # trn_size = 30
    # trn_size = 1
    # val_size = int(0.05 * len(dataset))
    # val_size = len(dataset) - trn_size
    # val_size = 5
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
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None).to(device)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
    # model = ResNet50NoAvgPool().to(device)
    # model.features = nn.Sequential(*list(model.features._modules.values())[:-1])
    # # model = ResNet18_noavgpool().to(device)
    # # model = ModifiedUNet
    
    
    num_ftrs = model.fc.in_features
    width = 256
    # print(f"input number of neurons in last layer is {25088, 2 * num_segments_per_line * width}")

    model.fc = nn.Linear(12800 , 2 * num_segments_per_line * width).to(device)

    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    
    # criterion = nn.MSELoss()
    criterion = HeatmapLoss()
    # criterion = nn.L1Loss()
    # criterion = RMSLELoss()
    rows = np.linspace(419, 50, num_segments_per_line).astype(int)

    for epoch in range(1, epochs + 1):
        print("Epoch {0}".format(epoch))
        model.train()
        train_loss = 0.0
        for i_batch, batch in enumerate(trn_loader):
            # print(f"Epoch {epoch}. Batch: {i_batch}")
            images = batch["image"]
            filenames = batch["filename"]
            # points = batch["left_lane_heatmap"].float()
            # create 1D gaussian for each point, so each point of lane is represented as 1x1280
            left_lane_heatmap = batch["left_lane_heatmap"].float()
            right_lane_heatmap = batch["right_lane_heatmap"].float()
            images, left_lane_heatmap, right_lane_heatmap = images.to(device), left_lane_heatmap.to(device), right_lane_heatmap.to(device)
            net_output = model(images).float()
            # for batch 8 - left lane heatmap's shape is (8, 1280 * 15 * 2)
            # TODO check!!!
            left_indices = torch.where(left_lane_heatmap.argmax(axis=2) > 2)
            # left_indices = torch.where(left_lane_heatmap > 2)
            right_indices = torch.where((right_lane_heatmap.argmax(axis=2) < 1277) & (right_lane_heatmap.argmax(axis=2) > 2))
            # print(f"left indices coords are {left_indices[0], left_indices[1]}")
            # print("---------")
            # indices = torch.where(points > 2)


            # plt.savefig("predicted_imgs/epoch_" + str(epoch) + "_" + str(random.randint(0, 900)) + ".png")
            # loss = criterion(net_output[indices], points[indices])
            # left_lane_nn_output.shape is width * 30 
            # print(net_output.shape)
            # TODO check if this works correctly for batch > 1
            try:
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



            if i_batch % 10 == 0:
                print('[TRN] Train epoch: {}, batch: {}\tLoss: {:.4f}'.format(
                    epoch, i_batch, loss.item()))
        print(f"training loss is {train_loss / len(trn_loader)}")
        train_losses.append(train_loss / len(trn_loader))

        model.eval()

        val_loss = 0.0
        with torch.no_grad():
            # if epoch < 25:
            #     continue
            # TODO switch to val_loader
            for i_batch, batch in enumerate(trn_loader):
                images = batch["image"]
                left_lane_heatmap = batch["left_lane_heatmap"].float()
                right_lane_heatmap = batch["right_lane_heatmap"].float()
                filenames = batch["filename"][0]
    
                images, left_lane_heatmap, right_lane_heatmap = images.to(device), left_lane_heatmap.to(device), right_lane_heatmap.to(device)
                net_output = model(images).float()
                left_lane_nn_output, right_lane_nn_output = np.split(net_output, 2, axis=1)
                ##########
                left_indices = torch.where(left_lane_heatmap.argmax(axis=2) > 2)
                right_indices = torch.where((right_lane_heatmap.argmax(axis=2) < 248) & (right_lane_heatmap.argmax(axis=2) > 2))
                left_lane_nn_output, right_lane_nn_output = np.split(net_output, 2, axis=1)
                try:
                    left_lane_nn_output = left_lane_nn_output.reshape(batch_sz, 20, width)
                    right_lane_nn_output = right_lane_nn_output.reshape(batch_sz, 20, width)
                except RuntimeError:
                    continue

                ###########


                loss = criterion(left_lane_nn_output[left_indices].flatten(), left_lane_heatmap[left_indices].flatten())
                loss += criterion(right_lane_nn_output[right_indices].flatten(), right_lane_heatmap[right_indices].flatten())

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
                for i in range(len(images)):
                    prediction = net_output[i].cpu()
                    # TODO why [1] index?
                    cur_left_indices = left_indices[1][np.where(left_indices[0].cpu() == i)].cpu()
                    cur_right_indices = right_indices[1][np.where(right_indices[0].cpu() == i)].cpu()

                    # true_left_lane = left_lane_heatmap[i]
                    # true_right_lane = right_lane_heatmap[i]
                    # true_left_lane = true_left_lane[i].reshape(15, width)
                    # true_right_lane = true_right_lane[i].reshape(15, width)
                    # left_indices = np.where(true_left_lane.argmax(axis=1) > 2)
                    # right_indices = np.where((true_right_lane.argmax(axis=1) > 2) & (true_right_lane.argmax(axis = 1) < 1277))
                    # label = points[i].cpu()
                    pred_left_lane, pred_right_lane = np.split(prediction, 2)
                    pred_left_lane = pred_left_lane.reshape(20, width)
                    pred_right_lane = pred_right_lane.reshape(20, width)
                    left_lane_pts = np.argmax(pred_left_lane, axis=1)
                    right_lane_pts = np.argmax(pred_right_lane, axis=1)
                    left_heatmap_mask = np.zeros((420, 256))
                    # TODO probably wont work
                    left_heatmap_mask[[rows], :] = pred_left_lane
                    right_heatmap_mask = np.zeros((420, 256))
                    right_heatmap_mask[[rows], :] = pred_right_lane
                    # print(filenames)
                    left_heatmap_mask = left_heatmap_mask.resize(420, 1280)
                    right_heatmap_mask = right_heatmap_mask.resize(420, 1280)
                    filename = filenames[i].split("/")[-1]
                    if epoch > 10:
                        plt.clf()
                        heatmap = plt.imshow(left_heatmap_mask, cmap='hot', interpolation='nearest')
                        plt.colorbar(heatmap)
                        # plt.show()
                        plt.savefig("heatmaps/" + directory + "/left_heatmap_epoch_" + str(epoch) + "_" + filename)
                        # ---
                        plt.clf()
                        heatmap = plt.imshow(right_heatmap_mask, cmap='hot', interpolation='nearest')
                        plt.colorbar(heatmap)
                        # plt.show()
                        
                        plt.savefig("heatmaps/" + directory + "/right_heatmap_epoch_" + str(epoch) + "_" + filename)


                    # cv2.imwrite("left_heatmap_val_dataset.png", left_heatmap_mask)
                    # print(f"shape of left lane pts is {left_lane_pts.shape}")
                    # TODO split array so every row has 1280 points
                    # TODO from those 1280 points -> argmax -> we have now corresponding prediction

                    # true_left_lane, true_right_lane = np.split(label, 2)
                    # left_indices = torch.where(left_lane_heatmap > 2)[0]
                    # right_indices = torch.where(right_lane_heatmap > 2)[0]
                    # print(left_indices)
                    # print()
                    # print(f"len of pred_left_lane is: {indices[0].shape}")
                    # print(f"len of true labels is {len(y)}")
                    # img_indices = indices[i].cpu()
                    # print(f"shape is {left_lane_heatmap.shape}")

                    if epoch > 0:
                        # plot_imgs(images[i], pred_left_lane, pred_right_lane, true_left_lane, true_right_lane, rows, indices, epoch)
                        plt.clf()
                        plt.plot(left_lane_pts[cur_left_indices] * 5, rows[cur_left_indices], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                        plt.plot(right_lane_pts[cur_right_indices] * 5, rows[cur_right_indices],  marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                        plt.plot(left_lane_heatmap[i][cur_left_indices].argmax(axis=1) * 5, rows[cur_left_indices], "b", right_lane_heatmap[i][cur_right_indices].argmax(axis=1) * 5, rows[cur_right_indices], "y")
                    # i_indices = indices.cpu()
                    # print(indices.size())
                    # return
                    # plt.plot(pred_left_lane[indices], rows[indices], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                    # plt.plot(pred_right_lane[indices], rows[indices],  marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                    # plt.plot(true_left_lane[indices], rows[indices], "b", true_right_lane[indices], rows[indices], "y")

                        plt.imshow((images[i].cpu().permute(1, 2, 0))) #.numpy().astype(np.uint8)))
                        plt.savefig("predicted_imgs/" + directory + "/epoch_" + str(epoch) + "_" + filename)

                        # plt.savefig("predicted_imgs/epoch_" + str(epoch) + "_" + filename)
                val_loss += loss.item()
            print(f"validation loss is {val_loss / len(val_loader)}")
            val_losses.append(val_loss / len(val_loader))
            plot_losses(epoch, train_losses, val_losses)

        torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--synthetic', action='store_true',
                            help='True if you want to train synthetic data')
        parser.add_argument('--pretrained', action='store_true',
                            help='True if you want to use pretrained model trained on synthetic dataset')
        args = parser.parse_args()
        timestamp = datetime.datetime.now()
        # dataset_length = 2000
        if args.synthetic:
        # dataset_length = 1000
        # train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length))
        # dataset_length = 5000
        # train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length))
        # dataset_length = 10000
        # train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length))
            dataset_length = 20
            train(args.synthetic, args.pretrained, dataset_length, "synthetic_" + str(dataset_length) + "_" + str(timestamp))
        else:
            dataset_length = 850
            directory = "real_world_" + str(dataset_length) + "_" + str(timestamp)
            if args.pretrained:
                directory = "real_world_" + "pretrained_" + str(dataset_length) + "_" + str(timestamp)
            train(args.synthetic, args.pretrained, dataset_length, directory)
        # train(args.synthetic, args.pretrained, dataset_length=dataset_length, directory=)
# TODO synthetic dataset!!!
# TODO predicting too much points - solved??
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