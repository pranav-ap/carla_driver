import random

import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches, orientation='horizontal'):
    # Convert images to RGB format if they are grayscale
    if len(img1.shape) == 2:  # Grayscale to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:  # Grayscale to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Add text using OpenCV

    img1 = put_text(img1, "top_center", "Previous Frame")
    img2 = put_text(img2, "top_center", "Current Frame")

    # Stack images based on the orientation
    if orientation == 'vertical':
        combined_img = stack_images(img1, img2, orientation='vertical')
    elif orientation == 'horizontal':
        combined_img = stack_images(img1, img2, orientation='horizontal')
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")

    # Draw matches on the combined image
    if orientation == 'vertical':
        height1 = img1.shape[0]
        adjusted_kp2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + height1, kp.size) for kp in kp2]
    else:
        width1 = img1.shape[1]
        adjusted_kp2 = [cv2.KeyPoint(kp.pt[0] + width1, kp.pt[1], kp.size) for kp in kp2]

    for match in matches:
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(adjusted_kp2[match.trainIdx].pt[0]), int(adjusted_kp2[match.trainIdx].pt[1]))

        # Draw lines between the matched keypoints
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(combined_img, pt1, pt2, color, 2)

        # Draw circles around keypoints
        cv2.circle(combined_img, pt1, 5, (255, 255, 255), -1)
        cv2.circle(combined_img, pt2, 5, (255, 255, 255), -1)

    return combined_img


def put_text(image, org, text, color=(0, 0, 255), font_scale=1, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    # to make sure it is writable
    image = image.copy()

    # Get the size of the text
    (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    w, h = image.shape[1], image.shape[0]

    place_h, place_w = org.split("_")

    org_w = 0
    org_h = 0

    # Calculate vertical position
    if place_h == "top":
        org_h = label_height + 10  # Add padding for top
    elif place_h == "bottom":
        org_h = h - label_height
    elif place_h == "center":
        org_h = h // 2 + label_height // 2

    # Calculate horizontal position
    if place_w == "left":
        org_w = 0
    elif place_w == "right":
        org_w = w - label_width
    elif place_w == "center":
        org_w = w // 2 - label_width // 2

    # Draw the text on the image using OpenCV
    cv2.putText(image, text, (org_w, org_h), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def stack_images(img1, img2, orientation='horizontal'):
    if orientation == 'vertical':
        combined_img = np.vstack((img1, img2))
    else:  # horizontal
        combined_img = np.hstack((img1, img2))

    return combined_img


