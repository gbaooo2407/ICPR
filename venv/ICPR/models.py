import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 66 * 66, 128)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_images, channels, height, width = x.size()
        x = x.view(batch_size * num_images, channels, height, width)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)

        # Đảm bảo đầu ra là [batch_size, num_classes]
        x = x.view(batch_size, num_images, -1)
        x = x.mean(dim=1)  # Tính trung bình trên các hình ảnh

        return x