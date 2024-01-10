#!/bin/bash

# Directory containing JPEG files
DIR="/content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/val"

# Count JPEG files
jpeg_count=$(find "$DIR" -name '*.JPG' -o -name '*.JPEG' | wc -l)

echo "Total number of JPEG files: $jpeg_count"
