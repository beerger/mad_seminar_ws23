

## Inference/Evaluation structure

For inference and evaluation, there are two files of importance: `brain_mri.ipynb`, and `anomaly_detector.py`. The AnomalyDetector class within the latter file is used for inference, and also during evaluation. For each image in the batch it will produce 400 patches (20 per side, overlapping) and feed it through the network, to compare the global and local features. It will then, using Inverse Distance Weighted interpolation produce and anomaly score map (pixel-level). This can then be used create anomaly heat maps etc.The mentioned Jupyter Notebook is used to create an instance of this AnomalyDetector class and use it for inference/evaluation.

## How to use

The notebook of interest is `brain_mri.ipynb` It consists of several sections:

1. `Fine-tune Local-Net`
2. `Joint training of Global-Net and DAD-head`
3. `Inference, and evaluating`

To run inference, and evaluate the model:

1. It is designed to run on Google Colab, thus there is no requirements.txt file available (nor needed)
2. Run the first 4 code blocks to clone the code from GitHub, download data, and get imports necessary to run
3. Scroll down to section 3 (`Inference, and evaluating`)
4. Run code blocks in order, where each comment above the cell will tell you a bit about what the code is for

## Note:

Ignore the code in the previous two sections in `brain_mri.ipynb`, as this is for fine-tuning/training, and is using specific paths to my Google Drive, which will thus result in an error if trying to run it