
# Used after extract_imagenet.sh failed to extract all tar files
# If directory exists and is empty, or directory doesn't exists it will unpack

# Navigate to the directory containing the individual class tar files
cd /content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/train

# Loop through tar files and extract if necessary
for tar_file in *.tar; do
    dir_name="${tar_file%.tar}"  # Removes the .tar extension to form the directory name

    # Check if the directory exists and is not empty
    if [ ! -d "$dir_name" ] || [ -z "$(ls -A "$dir_name")" ]; then
        echo "Extracting $tar_file..."
        mkdir -p "$dir_name"
        tar -xvf "$tar_file" -C "$dir_name"
    else
        echo "Directory for $tar_file already exists and is not empty. Skipping."
    fi
done
