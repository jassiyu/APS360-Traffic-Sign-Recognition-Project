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

def train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss for each epoch
    train_losses = []
    val_losses = []

    # Lists to store accuracy for each epoch
    train_acc = []
    val_acc = []
    
    # Training loop
    for epoch in range(num_epochs):

        # --- Training Phase ---
        model.train() #Set the model to training mode
        train_loss = 0.0
        correct_train = 0

        for imgs, labels in train_loader:
            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################
            #print(f"Processing batch...")
            
            optimizer.zero_grad()         # a clean up step for PyTorch
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter

            # Accumulate training loss and correct predictions
            train_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(out, 1)
            correct_train += (predicted == labels).sum().item()

        # Average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / len(train_loader.dataset)
        train_losses.append(avg_train_loss)  # Save train loss for this epoch
        train_acc.append(train_accuracy)  # Save train accuracy for this epoch

        # --- Validation Phase ---
        torch.save(model.state_dict(), data_dir +"model_lr{0}_epoch{1}.pth".format(learning_rate,epoch))
        model.eval()
        val_loss = 0.0
        correct_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                #############################################
                #To Enable GPU Usage
                if use_cuda and torch.cuda.is_available():
                  imgs = imgs.cuda()
                  labels = labels.cuda()
                #############################################

                # Forward pass
                out = model(imgs)
                loss = criterion(out, labels)

                # Accumulate validation loss and correct predictions
                val_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(out, 1)
                correct_val += (predicted == labels).sum().item()

        # Average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_val / len(val_loader.dataset)
        val_losses.append(avg_val_loss)  # Save validation loss for this epoch
        val_accuracy.append(val_accuracy)  # Save validation accuracy for this epoch

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    return train_losses, val_losses, train_acc, val_acc  # Return lists for plotting


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

  # Load the training dataset
  train_dataset = datasets.ImageFolder(data_dir + '/truncated_train', transform=train_transform)

  # Load the validation dataset
  val_dataset = datasets.ImageFolder(data_dir + '/truncated_val', transform=val_transform)

  # Number of samples in the training and validation datasets
  print(f'Number of training samples: {len(train_dataset)}')
  print(f'Number of validation samples: {len(val_dataset)}')

  batch_size = 32
  # Create data loaders for training and validation datasets
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  use_cuda = True
  Newmodel = LeNet5(num_classes=43) # GTSRB has 43 classes

  if use_cuda and torch.cuda.is_available():
    Newmodel.cuda()
    print('CUDA is available!  Training on GPU ...')
  else:
    print('CUDA is not available.  Training on CPU ...')

  train_losses, val_losses, train_acc, val_acc = train(Newmodel, train_loader, val_loader,num_epochs=40,learning_rate=0.001)

  print('Training completed!')

  # Create subplots for side-by-side plots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
  
  # Plot loss on the first subplot
  ax1.plot(train_losses, label='Training Loss')
  ax1.plot(val_losses, label='Validation Loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.set_title('Training and Validation Loss')
  ax1.legend()
  
  # Plot accuracy on the second subplot
  ax2.plot(train_acc, label='Training Accuracy')
  ax2.plot(val_acc, label='Validation Accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.set_title('Training and Validation Accuracy')
  ax2.legend()

  # Show the plots
  plt.tight_layout()
  plt.show()
  """

  #Test
  test_dataset = GTSRBTestDataset(csv_file=data_dir +'Test.csv', root_dir=data_dir,transform=val_transform, label_mapping=label_mapping)
  print('DatasetCreated.')
  num_samples = 500
  indices = list(range(len(test_dataset)))
  random_sampled_indices = random.sample(indices, num_samples)
  test_subset = Subset(test_dataset, random_sampled_indices)
  # Create DataLoader for the subset
  test_loader = DataLoader(dataset=test_subset, batch_size=32, shuffle=True)
  print(f'Number of test samples: {len(test_subset)}')
  Newmodel.load_state_dict(torch.load(data_dir +'model_lr0.001_epoch39.pth',weights_only=True))
  evaluate(Newmodel,test_loader)
  """
