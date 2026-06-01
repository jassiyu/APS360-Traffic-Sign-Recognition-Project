import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import Model

save_dir = 'D:\\Study\\UT\\APS-360\\Project\\'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")

def visualize_predictions(model, data_loader, class_names, num_images=9):
        model.eval()
        images, labels = next(iter(data_loader))
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)

        plt.figure(figsize=(12, 12))
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            img = TF.to_pil_image(images[i])  # Convert tensor to PIL image for display
            plt.imshow(img)
            ax.set_title(f"Pred: {class_names[preds[i]]}, True: {class_names[labels[i]]}")
            plt.axis("off")
        plt.show()

def visualize_misclassified(model, data_loader, class_names, num_images=9):
    model.eval()  # Set the model to evaluation mode
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    # Loop through the data loader to find misclassifications
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in data_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

            # Identify misclassified images
            for i in range(len(labels)):
                if preds[i] != labels[i]:  # If prediction doesn't match the true label
                    misclassified_images.append(images[i])  # Store the misclassified image
                    misclassified_labels.append(labels[i])  # Store the true label
                    misclassified_preds.append(preds[i])  # Store the predicted label

    # Plot the misclassified images
    plt.figure(figsize=(12, 12))
    num_misclassified = min(num_images, len(misclassified_images))  # Ensure we don't exceed available misclassified images
    for i in range(num_misclassified):
        ax = plt.subplot(3, 3, i + 1)  # 3x3 grid for displaying images
        img = TF.to_pil_image(misclassified_images[i])  # Convert tensor to PIL image for display
        plt.imshow(img)
        ax.set_title(f"Pred: {class_names[misclassified_preds[i]]}, True: {class_names[misclassified_labels[i]]}")
        plt.axis("off")

    plt.show()

# Function to load datasets (make sure to customize paths and transformations)
def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=save_dir+'/truncated_train/', transform=transform)
    val_dataset = datasets.ImageFolder(root=save_dir+'/truncated_val/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_dataset.classes


# Training function with optimizations
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:  # Enable mixed precision only if scaler is available
                with torch.amp.autocast('cuda'):  # Updated to use 'cuda' string
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        # Save model state after each epoch
        model_save_path = f'{save_dir}/model_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, model_save_path)
        print(f'Model saved to {model_save_path}')

    # Plotting training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


############# import test data set ####################
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import random


# Label mapping for GTSRB dataset
default_label_mapping = {
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

default_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Dataset class
class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=default_transform, use_label_names=False, label_mapping=default_label_mapping):
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


# Define transformations

# Load the dataset with label names



# # Custom dataset for the test set
# class TestDataset(data.Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         img_path = self.data_frame.iloc[idx]['Path']  # Get image path from 'Path' column
#         image = Image.open(img_path).convert('RGB')  # Open image

#         if self.transform:
#             image = self.transform(image)

#         label = self.data_frame.iloc[idx]['ClassId']  # Get label from 'ClassId' column
#         return image, label

# # Set up transformations
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # Adjust this size based on your model's requirements
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Create test dataset and loader
# csv_file_path = '/content/drive/MyDrive/APS360/Project/Test.csv'  # Update with your CSV file path
# test_dataset = TestDataset(csv_file=csv_file_path, transform=transform)
# test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)  # Adjust batch size as needed

def load_data(batch_size=32):

    train_dataset = datasets.ImageFolder(root=save_dir+'/truncated_train/', transform=default_transform)
    val_dataset = datasets.ImageFolder(root=save_dir+'/truncated_val/', transform=default_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, train_dataset.classes

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in test_loader:
            # Move inputs and labels to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class

            total += labels.size(0)  # Increment the total count
            correct += (predicted == labels).sum().item()  # Increment the correct count

    test_accuracy = 100 * correct / total
    top_1_error_rate = 100 - test_accuracy

    return test_accuracy, top_1_error_rate


if __name__ == '__main__':
    print("Train")