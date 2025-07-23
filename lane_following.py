import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_lane_center(lanes, img_height, img_width):
    """
    Finds the center intercept and slope of the closest lane.
    The "closest" lane is assumed to be the one lowest in the image frame.
    """
    if not lanes:
        return None, None

    # Find the lane lowest in the image (max average y-coordinate)
    closest_lane = None
    max_avg_y = -1

    for lane in lanes:
        line1, line2 = lane[0][0], lane[1][0]
        avg_y = (line1[1] + line1[3] + line2[1] + line2[3]) / 4
        if avg_y > max_avg_y:
            max_avg_y = avg_y
            closest_lane = lane

    if closest_lane is None:
        return None, None
    
    # Calculate the center line of the closest lane
    l1, l2 = closest_lane[0][0], closest_lane[1][0]
    p1_center = ((l1[0] + l2[0]) / 2, (l1[1] + l2[1]) / 2)
    p2_center = ((l1[2] + l2[2]) / 2, (l1[3] + l2[3]) / 2)

    # Calculate slope of the center line
    dx = p2_center[0] - p1_center[0]
    dy = p2_center[1] - p1_center[1]
    
    if abs(dx) < 1e-6:  # Nearly vertical lane center
        center_slope = float('inf')
        center_intercept = p1_center[0]
    else:
        center_slope = dy / dx
        # Calculate the y-intercept 'b' from y = mx + b
        y_intercept = p1_center[1] - center_slope * p1_center[0]
        
        # Calculate the horizontal intercept (x-value) at the bottom of the image
        if abs(center_slope) < 1e-6:  # Nearly horizontal line
            center_intercept = img_width / 2  # Default to center
        else:
            center_intercept = (img_height - 1 - y_intercept) / center_slope

    return center_intercept, center_slope

def recommend_direction(center_intercept, img_width):
    """
    Recommends a direction based on the lane's center position.
    """
    if center_intercept is None:
        return "No lane detected"

    image_center_x = img_width / 2
    # Define a tolerance band in the middle of the image
    tolerance = img_width * 0.1  # Reduced tolerance for more responsive steering

    if center_intercept < image_center_x - tolerance:
        return "Steer Left ⬅️"
    elif center_intercept > image_center_x + tolerance:
        return "Steer Right ➡️"
    else:
        return "Go Forward ⬆️"