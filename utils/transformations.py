import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import cv2

# Rotation vector to quaternion
def rotation2Quat(rotvec):
    """Convert rotation vector to quaternion.
    
    :param rotvec: Numpy array of shape (3,) representing the rotation vector.
    :return: Quaternion object representing the rotation."""
    return Quaternion(matrix=cv2.Rodrigues(np.array(rotvec))[0]).elements

def reorderList(y, matching_matrix):
    y_reordered = []
    for i in matching_matrix:
        y_reordered.append(y[i])
    return np.array(y_reordered)

def heatmaps2Keyp(heatmaps, target_size=[400,640]):
    """
    Convert heatmaps to scaled keypoints coordinates (x, y) based on the target image size.

    :param heatmaps: List of heatmaps, each corresponding to a keypoint.
    :param target_size: Tuple (target_width, target_height) for scaling the coordinates.
    :return: List of tuples (x, y,v) representing the coordinates of each keypoint, scaled to the target size.
    """
    target_height, target_width = target_size
    coordinates = []
    for heatmap in heatmaps:
        # Find the maximum location in the heatmap
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Scale the coordinates
        scale_x = target_width / heatmap.shape[1]
        scale_y = target_height / heatmap.shape[0]
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)

        coordinates.append(x_scaled)
        coordinates.append(y_scaled)
        coordinates.append(2) # Set visibility to 2 (visible) to match the COCO format
    
    return coordinates


def extractCoord(keypoints,input_size=[400,640], target_size=[400,640]):
    """Extract (x, y) coordinates from keypoints list in format [x1, y1, v1, x2, y2, v2, ...]."""
    if input_size is None or target_size is None:
        raise ValueError("Input and target size must be provided.")
    else: 
        scale_y = target_size[0] / input_size[0]
        scale_x = target_size[1] / input_size[1]
        coordinates = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            coordinates.append((x_scaled, y_scaled))
    return coordinates

def scaleCoord(keypoints, input_size=[400,640], target_size=[400,640]):
    """Scale (x, y) coordinates from keypoints list in format [[x1,y1],[x2,y2]...]."""
    if input_size is None or target_size is None:
        raise ValueError("Input and target size must be provided.")
    else: 
        scale_y = target_size[0] / input_size[0]
        scale_x = target_size[1] / input_size[1]
        coordinates = []
        for x, y in keypoints:
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            coordinates.append((x_scaled, y_scaled))
    return coordinates

    

