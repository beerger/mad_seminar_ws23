#!/bin/bash

# Base directory containing subfolders
DIR="/content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/train"

# Initialize count
total_jpeg_count=0

# Loop through all subdirectories and count JPEG files
for sub_dir in "$DIR"/n*/; do
    echo "Counting in $sub_dir"
    sub_dir_count=$(find "$sub_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
    echo "$sub_dir_count JPEG files in $sub_dir"
    
    # Add to total count
    total_jpeg_count=$((total_jpeg_count + sub_dir_count))
done

echo "Total number of JPEG files in all subdirectories: $total_jpeg_count"
