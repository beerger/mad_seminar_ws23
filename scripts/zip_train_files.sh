# Navigate to the dataset directory
%cd /content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/train

# Create a zip file containing only directories (excluding .tar files)
!find . -type d -name 'n*' -exec zip -r /content/drive/MyDrive/AnomalyDetection/Datasets/ImageNet/train.zip {} +
