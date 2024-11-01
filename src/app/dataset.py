import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):
        """
        Initialize the dataset with lists of image and mask paths.
        
        Parameters:
            images_path (list): List of paths to the input images.
            masks_path (list): List of paths to the corresponding masks.
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding mask, preprocess them, and convert to tensors.

        Parameters:
            index (int): Index of the sample to retrieve.
        
        Returns:
            image (torch.Tensor): Preprocessed image tensor.
            mask (torch.Tensor): Preprocessed mask tensor.
        """
        # Reading and normalizing the image
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0  # Normalize to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # Convert to channel-first format
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        # Reading and normalizing the mask
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.n_samples
