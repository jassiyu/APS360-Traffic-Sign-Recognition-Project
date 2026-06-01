# APS360 Traffic Sign Recognition Project
With the development of autonomous driving and Advanced Driver Assistance Systems (ADAS), Traffic Sign Recognition (TSR) has become an essential technology to ensure road safety and compliance with traffic regulations. The goal of this project is to develop a real-time TSR system using Convolutional Neural Networks (CNN) to recognize traffic signs in real-world environments accurately. The model uses deep learning techniques to capture the basic features of traffic signs - shape and color, which will effectively handle a wide range of challenging conditions. By utilizing the German Traffic Sign Recognition Benchmark (GTSRB) dataset, our model will be trained to provide high accuracy in applications. This project demonstrates how TSR can help reduce the number of errors made by humans and help improve decision-making performance, thus making transportation safer and more reliable. We will evaluate the performance of the model by comparing it to the baseline model LeNet-5, demonstrating the improvements in the accuracy and efficiency of modern CNN architectures.


Edit save_directory in "train.py "before running.
This repository is now organized around the final notebook reference in `notebooks/360project.ipynb`.

Main code layout:
- `360project.py`: main entry point for the cleaned primary-model workflow
- `src/aps360_project/`: reusable project modules
- `scripts/`: focused training and evaluation entry scripts
- `legacy/`: old Colab, baseline, and duplicate scripts kept for reference only
- `deliverables/`: reports and final figures
- `archives/`: zip archives and template files

Typical commands:
- `python 360project.py`
- `python scripts/train_primary.py`
- `python scripts/evaluate_primary.py --checkpoint outputs/checkpoints/primary_model_epoch_030.pth`
- `python scripts/train_baseline.py`

Default dataset paths still point to:
- `truncated_train/`
- `truncated_val/`
- `Test.csv`
- `Test/`
