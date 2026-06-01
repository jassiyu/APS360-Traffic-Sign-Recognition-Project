from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF


def plot_history(history: dict[str, list[float]], output_path: Path | None = None) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.show()


def visualize_predictions(
    model: torch.nn.Module,
    data_loader,
    class_names: Sequence[str],
    device: torch.device,
    num_images: int = 9,
) -> None:
    model.eval()
    images, labels = next(iter(data_loader))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 12))
    for index in range(min(num_images, len(images))):
        axis = plt.subplot(3, 3, index + 1)
        image = TF.to_pil_image(images[index])
        plt.imshow(image)
        predicted_index = int(preds[index].cpu())
        true_index = int(labels[index])
        axis.set_title(f"Pred: {class_names[predicted_index]}, True: {class_names[true_index]}")
        plt.axis("off")
    plt.show()


def visualize_misclassified(
    model: torch.nn.Module,
    data_loader,
    class_names: Sequence[str],
    device: torch.device,
    num_images: int = 9,
) -> None:
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)

            for index in range(len(labels)):
                if preds[index].cpu() != labels[index]:
                    misclassified_images.append(images[index])
                    misclassified_labels.append(labels[index])
                    misclassified_preds.append(preds[index].cpu())

    plt.figure(figsize=(12, 12))
    for index in range(min(num_images, len(misclassified_images))):
        axis = plt.subplot(3, 3, index + 1)
        image = TF.to_pil_image(misclassified_images[index])
        plt.imshow(image)
        axis.set_title(
            "Pred: "
            f"{class_names[int(misclassified_preds[index])]}, "
            f"True: {class_names[int(misclassified_labels[index])]}"
        )
        plt.axis("off")
    plt.show()
