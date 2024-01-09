#!/bin/bash

# Set the current directory to where the tar files are located
cd /content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/TrainValTar

IMAGENET_DIR="/content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet"

# Create train and val directories inside the imagenet directory
mkdir -p "$IMAGENET_DIR/train"
mkdir -p "$IMAGENET_DIR/val"

# Extract training data
tar -xvf ILSVRC2012_img_train.tar -C "$IMAGENET_DIR/train"

# Navigate to the training directory
cd "$IMAGENET_DIR/train"

# Extract each class' tar file
find . -name "*.tar" | while read NAME ; do 
    DIR_NAME=$(basename "$NAME" .tar)
    mkdir -p "$DIR_NAME"
    tar -xvf "$NAME" -C "$DIR_NAME"
    #rm "$NAME"
done

# Navigate back to the directory containing the validation data tar file
cd /content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/TrainValTar

# Extract validation data
tar -xvf ILSVRC2012_img_val.tar -C "$IMAGENET_DIR/val"

# Navigate to the validation directory
cd "$IMAGENET_DIR/val"

# You will need a separate script to organize the validation images, which can be downloaded and run
# Download the script to organize validation images
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
