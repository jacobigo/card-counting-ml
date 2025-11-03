# %% [markdown]
# ### Simple CNN to classify playing cards
# by Jacob Igo

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(torch.cuda.is_available())
print(f"Current GPU device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# %% [markdown]
# Creating the Model Architecture

# %%
"""
class CardNet(nn.Module):
    def __init__(self, num_classes=14):
        super(CardNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""


# %% [markdown]
# Creating the Card Dataset Class

# %%
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image


class CardDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.value_mapping = self._extract_card_value()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image, old_label = self.data[index]
        card_name = self.data.classes[old_label]
        value_label = self.value_mapping[card_name]
        return image, value_label
    
    def _extract_card_value(self):
        value_map = {}
        value_names = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven', 
                      'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'joker']
        
        for class_name in self.data.classes:
            class_lower = class_name.lower()
            for i, value in enumerate(value_names):
                if value in class_lower:
                    value_map[class_name] = i
                    break
            else:
                # Handle edge cases or unknown cards
                value_map[class_name] = 13  # Default to joker category
                
        return value_map
    
    @property
    def classes(self):
        return ['ace', 'two', 'three', 'four', 'five', 'six', 'seven', 
                'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'joker']

train_dir = r'C:\Users\jacob\card-counting-dataset\archive_1\train'
train_set = CardDataset(data_dir = train_dir)

target_to_class = {i: name for i, name in enumerate(train_set.classes)}
print(target_to_class)


# %% [markdown]
# Creating the DataLoaders

# %%
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dir = r'C:\Users\jacob\card-counting-dataset\archive_1\train'
val_dir = r'C:\Users\jacob\card-counting-dataset\archive_1\valid'
test_dir = r'C:\Users\jacob\card-counting-dataset\archive_1\test'

train_set = CardDataset(train_dir, transform=transform_train)
val_set = CardDataset(val_dir, transform=transform_val)
test_set = CardDataset(test_dir, transform=transform_val)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# %%
import matplotlib.pyplot as plt
import numpy as np
import torchvision



# %% [markdown]
# Model Training

# %%

if __name__ == '__main__':


    #model = CardNet()

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 14)

    model.train()

    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    best_acc = 0

    EPOCHS = 7
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00075)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(0, EPOCHS):
        model.train()
        total_samples_t, correct_predictions_t, running_loss_t = 0.0, 0.0, 0.0
        for i, data in enumerate(train_loader):
            image, label = data

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs = model(image)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, label)

            loss.backward()

            optimizer.step()

            running_loss_t += loss.item()
            total_samples_t += label.size(0)
            correct_predictions_t += (predicted == label).sum().item()

        model.eval()
        with torch.no_grad():
            total_samples_v, correct_predictions_v, running_loss_v = 0.0, 0.0, 0.0
            for i, data in enumerate(val_loader):

                image, label = data
                
                image = image.to(device)
                label = label.to(device)

                outputs = model(image)

                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, label)

                running_loss_v += loss.item()
                total_samples_v += label.size(0)
                correct_predictions_v += (predicted == label).sum().item()
        
        epoch_avg_loss_t = running_loss_t / len(train_loader)
        epoch_avg_loss_v = running_loss_v / len(val_loader)
        accuracy_t = 100 * correct_predictions_t / total_samples_t
        accuracy_v = 100 * correct_predictions_v / total_samples_v
        best_acc = max(accuracy_v, best_acc)
        train_acc.append(accuracy_t)
        val_acc.append(accuracy_v)
        train_loss.append(epoch_avg_loss_t)
        val_loss.append(epoch_avg_loss_v)
        print(f"Epoch: {epoch+1}/{EPOCHS}, Train avg loss:  {epoch_avg_loss_t:.4f}, Train avg acc: {accuracy_t:.4f} || val avg loss: {epoch_avg_loss_v:.4f}, val avg acc: {accuracy_v:.4f}")


    # %% [markdown]
    # Testing of the model

    # %%
    model.eval()
    with torch.no_grad():
        test_acc, test_count, test_pred = 0.0, 0.0, 0.0
        for i, data in enumerate(test_loader):
            image, label = data

            image = image.to(device)
            label = label.to(device)

            outputs = model(image)

            _, predicted = torch.max(outputs.data, 1)

            test_count += label.size(0)
            test_pred += (predicted == label).sum().item()

    accuracy = 100 * test_pred / test_count
    print(f"Accuracy: {accuracy}")


    if best_acc >= 80 and accuracy >= 80:
        torch.save(model.state_dict(), 'cardnet_1.pth')
    else:
        print(f"didn't save, best_acc was {best_acc} and test accuracy was {accuracy}")
    


        
# %% [markdown]
# Saving the Model



