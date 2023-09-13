import cv2
import numpy as np
import glob
# from google.colab.patches import cv2_imshow
import os
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

def get_rotated_image(unrotated_image):

  rotated_image = cv2.rotate(unrotated_image, cv2.ROTATE_90_CLOCKWISE)
  return rotated_image


def detect_car_shadow(original_image, segmentation_mask):
    # Convert the images to grayscale if they are not already.
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    if len(segmentation_mask.shape) == 3:
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)

    # Ensure both images have the same size.
    if original_image.shape != segmentation_mask.shape:
        raise ValueError("Original image and segmentation mask must have the same dimensions.")

    # Apply a Gaussian blur to the original image to reduce noise.
    original_image_blur = cv2.GaussianBlur(original_image, (5, 5), 0)

    # Calculate the absolute difference between the original image and its segmentation mask.
    diff = cv2.absdiff(original_image_blur, segmentation_mask)
    # cv2_imshow(diff)
    # Threshold the difference image to obtain a binary mask of potential shadows.
    threshold_value = 20  # Adjust this threshold as needed.
    _, shadow_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)


    # Use a morphological operation (e.g., dilation) to connect the car with its shadow.
    kernel = np.ones((5, 5), np.uint8)

    shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

    car_shadow_mask = 255-cv2.subtract(shadow_mask, segmentation_mask)

    kernel = np.ones((9,9), np.uint8)
    closing = cv2.morphologyEx(car_shadow_mask, cv2.MORPH_CLOSE, kernel)
    mask_with_shadow = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return mask_with_shadow


def add_car_virtualbg(image_1,car_mask,image_2):
    # print("image_1")
    # cv2_imshow(image_1)
    # print("car_mask")
    # cv2_imshow(car_mask)
    
    image_2 = cv2.resize(image_2, (1920, 1080), interpolation=cv2.INTER_AREA)
    # print("image_2")
    # cv2_imshow(image_2)
    car_shadow_mask = detect_car_shadow(image_1, car_mask)
    car_mask_bgr = cv2.cvtColor(car_shadow_mask, cv2.COLOR_GRAY2BGR)
    car_mask_gray = cv2.cvtColor(car_mask_bgr, cv2.COLOR_BGR2GRAY)
    # print("car_mask_gray")
    # cv2_imshow(car_mask_gray)
    contours, hierarchy = cv2.findContours(car_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    car = cv2.bitwise_and(image_1, car_mask_bgr)

    # Extract car image using bounding box
    car_img = car[y:y+h, x:x+w]
    car_mask_a = car_mask_bgr[y:y+h, x:x+w]
    # print("car_img")
    # cv2_imshow(car_img)
    # print("car_mask_a")
    # cv2_imshow(car_mask_a)

    x_position , y_position = 0,image_2.shape[0]-car_img.shape[0]

    ratio = car_img.shape[0]/car_img.shape[1]
    new_width  = int(image_2.shape[1]*0.70)
    new_height = int(new_width*ratio)

    resized_car = cv2.resize(car_img, (new_width, new_height))
    resized_mask = cv2.resize(car_mask_a, (new_width, new_height))

    x_position = (image_2.shape[1] - new_width)//2
    y_position = image_2.shape[0]-resized_car.shape[0]-int(image_2.shape[0]*0.05)

    new_x, new_y = (x_position, y_position)

    result = np.zeros_like(image_2)
    result_mask = np.zeros_like(image_2)

    result[new_y:new_y + new_height, new_x:new_x + new_width] = resized_car
    result_mask[new_y:new_y + new_height, new_x:new_x + new_width] = resized_mask

    background_mask = 255 - result_mask
    background = cv2.bitwise_and(image_2, background_mask)
    car_virtual = cv2.add(result, background)

    return car_virtual


def main():
    parser = argparse.ArgumentParser(description="Your description here")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the mask directory")
    parser.add_argument("--background_path", type=str, required=True, help="Path to the background image")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    background_path = args.background_path
    save_dir = args.save_dir

    folder_name = save_dir

# Check if the folder exists
    if not os.path.exists(folder_name):
        # If it doesn't exist, create it
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    image_files = sorted(glob.glob(image_dir + '/*'))
    mask_files = sorted(glob.glob(mask_dir + '/*'))
    background = cv2.imread(background_path)
    for i in range(0,len(image_files)):
        input_image = cv2.imread(image_files[i])
        mask_image = cv2.imread(mask_files[i])
        orientation = detect_car_orientation(mask_image)
        if orientation == "Vertical":
            input_image = get_rotated_image(input_image)
            mask_image = get_rotated_image(mask_image)
        car_virtual = add_car_virtualbg(input_image,mask_image,background)

        img_name = image_files[i].split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        prediction_dir = save_dir+os.sep+imidx+".png"
        cv2.imwrite(prediction_dir, car_virtual)


if __name__ == "__main__":
    main()