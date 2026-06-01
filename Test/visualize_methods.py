from primy_mode import *
# Visualization function
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