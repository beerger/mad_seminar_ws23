import torch
from torch import Tensor
from torchvision import transforms
import numpy as np
from model.iad import iad_head
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter

class AnomalyDetector:
    def __init__(self, local_net, global_net, dad_head, none=False):
        """
        Initialize the anomaly detector with Local-Net, Global-Net, IAD-Head, and DAD-Head.
        
        Args:
        - local_net: Trained Local-Net model for extracting local features.
        - global_net: Trained Global-Net model for extracting global features.
        - iad_head: Function or model to compute the IAD score.
        - dad_head: Function or model to compute the DAD score.
        - lambda_s: Weighting factor to combine IAD and DAD scores.
        """
        self.local_net = local_net
        self.global_net = global_net
        self.dad_head = dad_head
        self.lambda_s = 0.8
        
        if not(none):
            self.local_net.eval()
            self.global_net.eval()
            self.dad_head.eval()

    def detect_anomalies(self, image: Tensor):
        """
        Detect anomalies in the given image.
        
        Args:
        - image: The input image tensor.
        
        Returns:
        - Anomaly score map indicating the presence of anomalies.
        """
        patches, binary_masks, patch_coords = self.create_patches_and_masks(image, return_coords=True)

        # Initialize lists to hold individual scores
        iad_scores = []
        dad_scores = []

        # Iterate over each patch to extract local features and compute scores
        for i in range(len(patches)):
            global_features, _ = self.global_net(image, binary_masks[i])
            local_features = self.local_net(patches[i])
            iad_score = iad_head(local_features, global_features)
            local_features_flat = torch.flatten(local_features, start_dim=1)
            global_features_flat = torch.flatten(global_features, start_dim=1)
            # Concatenate features for DAD-head input
            combined_features = torch.cat((local_features_flat, global_features_flat), dim=1)
            dad_score = 1 - self.dad_head.infer(combined_features)
            iad_scores.append(iad_score)
            dad_scores.append(dad_score)

        # Combine IAD and DAD scores to create a final score for each patch
        final_scores = [self.lambda_s * iad + (1 - self.lambda_s) * dad for iad, dad in zip(iad_scores, dad_scores)]
        # Generate an anomaly map based on the final scores
        anomaly_map = self.generate_anomaly_map(final_scores, image.shape, patch_coords)

        return anomaly_map
    
    def detect_anomalies_batch(self, images: Tensor):
        """
        Detect anomalies in a batch of images.

        Args:
        - images: A batch of input image tensors.

        Returns:
        - A list of anomaly score maps, one for each image.
        """
        batch_anomaly_maps = []

        for image in images:
            # unsqueeze to add a 4th dimension, expected by the network's forward pass
            anomaly_map = self.detect_anomalies(image.unsqueeze(0))
            batch_anomaly_maps.append(anomaly_map)

        return batch_anomaly_maps
    
    def evaluate_performance(self):
        pass

    def visualize_anomaly(self, image: Tensor, anomaly_map: np.ndarray, save_path=None, numpy_save_path=None, alpha=0.48, sigma=5):
        """
        Visualize the anomaly by blending the anomaly map with the original image.

        Args:
        - image: The original image tensor.
        - anomaly_map: The anomaly map as a numpy array.
        - save_path: Optional path to save the visualized image.
        - numpy_save_path: Optional path to save numpy array of blended image
        - alpha: Transparency factor for blending.
        - sigma: Sigma value for Gaussian blur.

        Returns:
        - None. Displays or displays and saves the blended image.
        """

        # Apply Gaussian blur to the anomaly map
        smoothed_anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)

        # Convert to numpy array and transpose to HWC format for image display
        image_np = image.cpu().detach().numpy().transpose(1, 2, 0)
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        # Normalize and apply colormap
        normalized_anomaly_map = Normalize(0, 1)((smoothed_anomaly_map - np.min(smoothed_anomaly_map)) / (np.max(smoothed_anomaly_map) - np.min(smoothed_anomaly_map)))
        
        # Apply a colormap to the normalized anomaly map
        colormap = plt.cm.jet
        normed_data = Normalize(0, 1)(normalized_anomaly_map)
        mapped_data = colormap(normed_data)
        
        # Convert the RGBA image to an RGB image
        mapped_data_rgb = (mapped_data[..., :3] * 255).astype(np.uint8)

        # Blend and display or save
        blended_image = Image.blend(image_pil.convert("RGBA"), Image.fromarray(mapped_data_rgb).convert("RGBA"), alpha=alpha)
        if save_path:
            blended_image.save(save_path)
        if numpy_save_path:
            blended_np = np.array(blended_image)
            np.save(numpy_save_path, blended_np)
        
        plt.imshow(blended_image)
        plt.axis('off')
        plt.show()

    def create_patches_and_masks(self, image, patch_size=33, patches_per_side=20, return_coords=False):

        image_size = 256
        step = (image_size - patch_size) / (patches_per_side - 1)
        # crop takes y, x, h, w
        crop_coords = []
        for j in range(patches_per_side):
          for i in range(patches_per_side):
            crop_coords.append((int(j*step), int(i*step), patch_size, patch_size))

        patches = []
        masks = []
        for coord in crop_coords:
            y, x, h, w = coord
            patch = transforms.functional.crop(image, y, x, h, w)
            patch = self.transform_local(patch)
            patches.append(patch)
            masks.append(self.generate_mask(coord))
        if return_coords:
            return patches, masks, crop_coords
        return patches, masks

    def generate_mask(self, crop_coordinates, mask_size=256):
        mask = torch.ones((1, mask_size, mask_size))
        y, x, h, w = crop_coordinates  # Unpack the crop coordinates
        mask[:,y:y+h, x:x+w] = 0
        return mask

    def transform_local(self, image):
        # Define transforms for Local-Net
        return transforms.Compose([
            # Mean and std from ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)


    def generate_anomaly_map(self, final_scores, image_shape, patch_coords, power=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Convert to PyTorch tensors and send to GPU
        final_scores = torch.tensor(final_scores).to(device)
        patch_coords = torch.tensor(patch_coords).to(device)
    
        height, width = image_shape[2], image_shape[3]
        xx, yy = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device))
        
        anomaly_map = torch.zeros((height, width), device=device)
        weight_map = torch.zeros_like(anomaly_map)
    
        for score, (y, x, h, w) in zip(final_scores, patch_coords):
            patch_center_x, patch_center_y = x + w // 2, y + h // 2
    
            # Compute distances for all pixels in parallel
            distance = torch.sqrt((xx - patch_center_x) ** 2 + (yy - patch_center_y) ** 2)
            weight = 1.0 / (distance ** power + 1e-6)  # Add a small epsilon to avoid division by zero
    
            anomaly_map += weight * score
            weight_map += weight
    
        # Normalize the anomaly map
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        anomaly_map /= weight_map

        return anomaly_map.cpu().numpy()  # Convert back to numpy array if needed
# This class would be used after all individual components have been instantiated and trained.
# Example usage:
# anomaly_detector = AnomalyDetector(local_net, global_net, iad_head, dad_head, lambda_s)
# anomaly_map = anomaly_detector.detect_anomalies(input_image)
