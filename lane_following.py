# lane_following.py

import numpy as np

def get_lane_center(lanes, img_height, img_width):
    """
    Finds the horizontal center of the detected lanes.
    It calculates the midpoint between the left and right lanes at the bottom of the image.
    """
    if lanes is None or len(lanes) == 0:
        return None, None

    bottom_points = []
    for lane in lanes:
        if lane is not None:
            x1, y1, x2, y2 = lane
            # Avoid division by zero for vertical lines
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Avoid division by zero for horizontal lines
            if abs(slope) < 1e-4:
                continue
            intercept = y1 - slope * x1
            x_bottom = (img_height - intercept) / slope
            bottom_points.append(x_bottom)

    if not bottom_points:
        return None, None
        
    center_intercept = np.mean(bottom_points)
    
    # Overall slope is not critical for this navigation logic
    center_slope = 0 
    
    return center_intercept, center_slope

def recommend_direction(center_intercept, img_width):
    """
    Recommends a direction based on the lane's center position relative to the image center.
    """
    if center_intercept is None:
        return "No lane detected, searching..."

    image_center_x = img_width / 2
    tolerance = img_width * 0.10

    if center_intercept < image_center_x - tolerance:
        return "Steer Left ⬅️"
    elif center_intercept > image_center_x + tolerance:
        return "Steer Right ➡️"
    else:
        return "Go Forward ⬆️"