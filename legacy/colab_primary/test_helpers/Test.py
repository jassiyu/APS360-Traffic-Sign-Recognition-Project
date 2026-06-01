import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
# Data Loaders
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision import datasets, transforms
from PIL import Image


# Label mapping for GTSRB dataset
label_mapping = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}


#################### import train data and validation ###################

# LeNet-5: Pytorch Implementation
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        # Convolutional layers for 3-channel RGB input
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(10 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def show_misclassified_images(images, true_labels, predicted_labels, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(5, 5, i + 1)
        img = images[i].permute(1, 2, 0)
        img = img * 0.5 + 0.5  # Unnormalize if normalized with (0.5, 0.5, 0.5)
        plt.imshow(img)
        plt.title(f'True: {true_labels[i]}, Pred: {predicted_labels[i]}')
        plt.axis('off')
    plt.show()

def evaluate(model, test_loader):
    # Lists to store misclassified images, labels, and predictions
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################
            outputs = model(imgs)            
            _, predicted = torch.max(outputs, 1)
            # Track misclassifications
            misclassified_mask = predicted != labels
            misclassified_images.extend(imgs[misclassified_mask].cpu())
            misclassified_labels.extend(labels[misclassified_mask].cpu())
            misclassified_predictions.extend(predicted[misclassified_mask].cpu())

            # Update correct predictions count
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test dataset: {accuracy:.2f}%')

    # Show a few misclassified images
    show_misclassified_images(misclassified_images, misclassified_labels, misclassified_predictions)


class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, use_label_names=False, label_mapping=None):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.root_dir = root_dir  # Root directory where images are stored
        self.transform = transform
        self.use_label_names = use_label_names  # Flag to use label names
        self.label_mapping = label_mapping  # Label mapping dictionary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 7].strip())

        #print(f"Loading image: {img_name}")

        image = Image.open(img_name)

        # Get the numerical label
        label = int(self.data.iloc[idx, 6])

        # If requested, use the label name instead of the numerical label
        if self.use_label_names and self.label_mapping:
            label = self.label_mapping.get(label, "Unknown")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label
  
if __name__ == '__main__':
  set_seed(1000)
  train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    # transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize images
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(25)
    #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
  ])

  # Define transformations for validation data
  val_transform = transforms.Compose([
      transforms.Resize((32, 32)),  # Resize images to 32x32
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
  ])

  # Set the path to the dataset
  data_dir = 'D:\\Study\\UT\\APS-360\\Project\\'

  # Plot the loss curves
  train_dataset = datasets.ImageFolder(data_dir + 'truncated_train', transform=train_transform)

  # Load the validation dataset
  val_dataset = datasets.ImageFolder(data_dir + 'truncated_val', transform=val_transform)

  # Number of samples in the training and validation datasets
  print(f'Number of training samples: {len(train_dataset)}')
  print(f'Number of validation samples: {len(val_dataset)}')

  batch_size = 1
  # Create data loaders for training and validation datasets
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
  use_cuda = True
  model = LeNet5(num_classes=43) # GTSRB has 43 classes

  if use_cuda and torch.cuda.is_available():
    model.cuda()
    print('CUDA is available!  Training on GPU ...')
  else:
    print('CUDA is not available.  Training on CPU ...')

  #Test
  test_dataset = GTSRBTestDataset(csv_file=data_dir +'Test.csv', root_dir=data_dir,transform=val_transform, label_mapping=label_mapping)
  print('DatasetCreated.')
  num_samples = 500
  indices = list(range(len(test_dataset)))
  random_sampled_indices = random.sample(indices, num_samples)
  test_subset = Subset(test_dataset, random_sampled_indices)
  # Create DataLoader for the subset
  test_loader = DataLoader(dataset=test_subset, batch_size=1, shuffle=True)
  print(f'Number of test samples: {len(test_subset)}')
  model.load_state_dict(torch.load(data_dir +'model_lr0.001_epoch39.pth',weights_only=True))
  evaluate(model,test_loader)