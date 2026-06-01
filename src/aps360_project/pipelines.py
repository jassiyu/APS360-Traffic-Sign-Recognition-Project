import argparse
from pathlib import Path
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from aps360_project.config import ProjectPaths, get_device
from aps360_project.data import (
    build_baseline_eval_transform,
    build_baseline_train_transform,
    build_primary_transform,
    load_imagefolder_split,
    load_primary_data,
    load_test_data,
)
from aps360_project.models import LeNet5, TrafficSignGoogLeNet
from aps360_project.training import evaluate_model, load_checkpoint, set_seed, train_model
from aps360_project.visualization import plot_history, visualize_misclassified, visualize_predictions


def _add_shared_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)


def _resolve_paths(args: argparse.Namespace) -> ProjectPaths:
    paths = ProjectPaths.from_repo_root()
    return ProjectPaths(
        root=paths.root,
        train_dir=args.train_dir or paths.train_dir,
        val_dir=args.val_dir or paths.val_dir,
        test_root=args.test_root or paths.test_root,
        test_csv=args.test_csv or paths.test_csv,
        checkpoints_dir=args.checkpoints_dir or paths.checkpoints_dir,
        figures_dir=args.figures_dir or paths.figures_dir,
    )


def _latest_checkpoint(checkpoints_dir: Path, prefix: str) -> Optional[Path]:
    matches = list(checkpoints_dir.glob(f"{prefix}_epoch_*.pth"))
    if not matches:
        return None

    def extract_epoch(path: Path) -> int:
        match = re.search(r"_epoch_(\d+)\.pth$", path.name)
        return int(match.group(1)) if match else -1

    return max(matches, key=extract_epoch)


def run_primary_training(args: argparse.Namespace) -> Optional[Path]:
    paths = _resolve_paths(args)
    paths.ensure_output_dirs()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, class_names = load_primary_data(
        paths=paths,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    model = TrafficSignGoogLeNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training primary model on {device}.")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=paths.checkpoints_dir,
        checkpoint_prefix="primary_model",
    )

    if not args.no_plots:
        plot_history(history, output_path=paths.figures_dir / "primary_training_history.png")
        visualize_predictions(model, val_loader, class_names, device=device)
        visualize_misclassified(model, val_loader, class_names, device=device, num_images=9)

    return _latest_checkpoint(paths.checkpoints_dir, "primary_model")


def run_primary_evaluation(args: argparse.Namespace) -> None:
    paths = _resolve_paths(args)
    device = get_device()

    if not paths.test_csv.exists():
        print(f"Skipping test evaluation because {paths.test_csv} does not exist.")
        return

    transform = build_primary_transform(image_size=args.image_size)
    test_loader = load_test_data(
        csv_file=paths.test_csv,
        root_dir=paths.test_root,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_size=args.test_samples,
        seed=args.seed,
    )

    train_loader, _, class_names = load_primary_data(
        paths=paths,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    model = TrafficSignGoogLeNet(num_classes=len(class_names)).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = _latest_checkpoint(paths.checkpoints_dir, "primary_model")

    if checkpoint_path is None:
        raise FileNotFoundError("No primary model checkpoint was found for evaluation.")

    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    accuracy, error_rate = evaluate_model(model, test_loader, device=device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Top-1 Error Rate: {error_rate:.2f}%")


def run_baseline_training(args: argparse.Namespace) -> Optional[Path]:
    paths = _resolve_paths(args)
    paths.ensure_output_dirs()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, class_names = load_imagefolder_split(
        train_dir=paths.train_dir,
        val_dir=paths.val_dir,
        train_transform=build_baseline_train_transform(image_size=args.image_size),
        val_transform=build_baseline_eval_transform(image_size=args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = LeNet5(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training baseline model on {device}.")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=paths.checkpoints_dir,
        checkpoint_prefix="baseline_model",
    )

    if not args.no_plots:
        plot_history(history, output_path=paths.figures_dir / "baseline_training_history.png")

    return _latest_checkpoint(paths.checkpoints_dir, "baseline_model")


def build_train_primary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final APS360 primary model.")
    _add_shared_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_evaluate_primary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the final APS360 primary model.")
    _add_shared_data_arguments(parser)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test-samples", type=int, default=1290)
    return parser


def build_train_baseline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the APS360 baseline model.")
    _add_shared_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_project_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the cleaned APS360 project workflow.")
    _add_shared_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=1290)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def train_primary_cli() -> None:
    args = build_train_primary_parser().parse_args()
    run_primary_training(args)


def evaluate_primary_cli() -> None:
    args = build_evaluate_primary_parser().parse_args()
    run_primary_evaluation(args)


def train_baseline_cli() -> None:
    args = build_train_baseline_parser().parse_args()
    run_baseline_training(args)


def run_project_cli() -> None:
    args = build_project_parser().parse_args()
    latest_checkpoint = args.checkpoint

    if not args.skip_train:
        latest_checkpoint = run_primary_training(args)

    if args.skip_test:
        return

    args.checkpoint = latest_checkpoint or args.checkpoint
    run_primary_evaluation(args)
