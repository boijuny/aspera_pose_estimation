import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from pycocotools.coco import COCO
import numpy as np


############################## 1. LOAD AND SAVE IMAGES ##############################
def loadImg(image_path,target_size=None):
    """Load an image from a file path."""
    cv2_image = cv2.imread(image_path)
    if target_size is not None:
        cv2_image = cv2.resize(cv2_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return cv2_image

def saveImg(image_path,image):
    """Save an image to a file path."""
    cv2.imwrite(image_path, image)
#####################################################################################

############################## 1. MANIPULATE IMAGES ANNOTATIONS ##############################
def loadImgAnns(file_name,coco=None):
    """
    Retrieve keypoints, their names, and the corresponding skeleton for a specific image by its file name from a COCO dataset.
    
    :param coco: The COCO object containing the annotations.
    :param file_name: The file name of the image for which to retrieve keypoints.
    :return: A dictionary with keypoints list, their names, and skeleton for all annotations related to the image or a message if no image is found.
    """
    #verify if coco is not None
    if coco is None:
        return "COCO object is not provided. You need to create a COCO object from pycocotools lib before calling this function. (e.g., coco = COCO(annotation_file)"
    
    # Get all images, then filter by file name to find the image ID
    all_images = coco.loadImgs(coco.getImgIds())
    target_image = [img for img in all_images if img['file_name'] == file_name]
    
    if not target_image:
        return "No image found with the given file name."
    
    # Extract the image ID
    image_id = target_image[0]['id']
    
    # Get all annotations for the image
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    # Extract skeletons from category metadata
    cat_ids = annotations[0]['category_id']
    categories = coco.loadCats(cat_ids)
    skeleton = categories[0]['skeleton']
    labels = categories[0]['keypoints']

    # add labels and skeleton to the annotations dictionary
    annotations[0]['skeleton'] = skeleton
    annotations[0]['keypoints_labels'] = labels

    return annotations[0]

def vizualizeAnns(image_path, keypoints=None, skeletons=None, names=None,heatmaps=None,target_size=[640,400], s_color='y'):
    """
    Plot keypoints and skeleton with PIL and Matplotlib on an image, where keypoints are provided in the format: [[x1, y1],[x2, y2], ...],
    and skeleton as pairs of indices indicating connected keypoints, with keypoint names displayed.

    :param image_path: Path to the image file.
    :param heatmaps: 2D numpy array representing the heatmap.
    :param target_size: Target size to resize the image before plotting the heatmap.
    :param keypoints: List of flat lists of keypoints in the format [x1, y1, v1, x2, y2, v2, ...].
    :param skeletons: List of lists containing pairs of indices (1-based) of keypoints which should be connected.
    :param keypoint_names: List of names corresponding to the keypoints, in the same order as they appear in keypoints.
    """
    # Load image using PIL
    if target_size:
        image = Image.open(image_path).resize((target_size[0], target_size[1]))
        plt.imshow(image)
        ax = plt.gca()
    else:
        image = Image.open(image_path)
        print(image.size)
        plt.imshow(image)
        ax = plt.gca()

    # Define distinct flashy colors for better visibility
    colors = [
        '#FF6666', '#FF3333', '#FF0000', '#CC0000',  # Red tints
        '#66FF66', '#33FF33', '#00FF00', '#00CC00',  # Green tints
        '#6666FF', '#3333FF', '#0000FF'              # Blue tints
    ]

    # Plot keypoints and display names
    if heatmaps:
        if len(heatmaps.shape) == 3:
            heatmap = np.sum(heatmaps, axis=0)  # Sum all heatmaps if multiple are provided
        else:
            heatmap = heatmaps # Use the single heatmap provided
        if target_size:
            heatmap = cv2.resize(heatmap, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
        if image_path:
            image = loadImg(image_path, target_size)
            plt.imshow(image)
        else:
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')

        for (x,y) in enumerate(keypoints):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v:
                plt.scatter(x, y, color='blue', s=10)
    
        plt.imshow(heatmap, alpha=0.5, cmap='hot', interpolation='nearest')
    
    if keypoints:
        for i, (x, y) in enumerate(keypoints):
            
            color = colors[i % len(colors)]
            ax.add_patch(Circle((x, y), radius=3, color=color, fill=True))
            if names:
                names = names[i]
                ax.text(x, y, names[i], color=color, fontsize=8, ha='left', va='center')
        # Plot skeleton
        if skeletons:
            for link in skeletons:
                start, end = link
                start -= 1  # Convert 1-based index to 0-based
                end -= 1
                if keypoints[start]!=(0,0) and keypoints[end]!=(0,0):  # Check if both keypoints are visible
                    # Draw a line between keypoints
                    x_values = [keypoints[start][0], keypoints[end][0]]
                    y_values = [keypoints[start][1], keypoints[end][1]]
                    ax.plot(x_values, y_values, f'{s_color}-', linewidth=0.7)  # Yellow line
    plt.axis('off')
    


