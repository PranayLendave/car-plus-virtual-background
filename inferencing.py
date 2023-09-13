import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse


def detect_car_orientation(masked_image):
    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "Orientation undetermined", None  # No car detected in the image

    # Assuming that the largest contour corresponds to the car
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit an oriented bounding box to the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    x = box[:,0]
    y = box[:,1]

    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)

    if width > height:
      orientation = "Horizontal"
    else:
      orientation = "Vertical"

    return orientation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your description here")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input image folder")
    parser.add_argument("--gt_folder", type=str, required=True, help="Path to the ground truth folder")
    parser.add_argument("--mask_folder", type=str, required=True, help="Path to the mask folder")

    args = parser.parse_args()

    input_folder = args.input_folder
    gt_folder = args.gt_folder
    mask_folder = args.mask_folder

    for filename in os.listdir(input_folder):
        name, ext = os.path.splitext(filename)

        input_path = os.path.join(input_folder, filename)
        gt_path = os.path.join(gt_folder, name + '.png')
        mask_path = os.path.join(mask_folder, name + '.png')

        input_img = plt.imread(input_path)
        gt_img = plt.imread(gt_path)
        mask_img = cv2.imread(mask_path)
        orientation = detect_car_orientation(mask_img)

        fig, axs = plt.subplots(1, 4, figsize=(10, 3))
        axs[0].imshow(input_img)
        axs[0].set_title('Input')
        axs[1].imshow(gt_img, cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(mask_img)
        axs[2].set_title('Prediction')
        # axs[3].axis('off')
        axs[3].text(0.5, 0.5, orientation,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14)
        axs[3].set_title('Orientation')
        plt.tight_layout()

        plt.show()
