import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
# Data Loaders
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

        # --- Validation Phase ---
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

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    return train_losses, val_losses  # Return losses for plotting


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
            out = model(imgs)            
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
  data_dir = 'D:\\Study\\UT\\APS-360\\Project'

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
  model = LeNet5(num_classes=43) # GTSRB has 43 classes

  if use_cuda and torch.cuda.is_available():
    model.cuda()
    print('CUDA is available!  Training on GPU ...')
  else:
    print('CUDA is not available.  Training on CPU ...')
      
  train_losses, val_losses = train(model, train_loader, val_loader,num_epochs=40,learning_rate=0.001)

  # Plot the loss curves
  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Training Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()
