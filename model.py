import torch.nn as nn
import torch.nn.functional as F

class ModifiedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(ModifiedUNet, self).__init__()

        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Regression head
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_channels)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x1 = x
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x2 = x
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x3 = x
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x4 = x
        x = self.dropout4(x)
        x = self.pool4(x)
         # Decoder
        x = self.upconv5(x)
        x = self.torch.cat([x, x4], dim=1)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))

        x = self.upconv6(x)
        x = self.torch.cat([x, x3], dim=1)
        x = self.F.relu(conv6_1(x))
        x = self.F.relu(conv6_2(x))

        x = self.upconv7(x)
        x = self.torch.cat([x, x2], dim=1)
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))

        # Regression head
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
