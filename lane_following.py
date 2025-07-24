import cv2
import numpy as np

def get_lane_center(lanes):
    """
    Gets the intercept and slope of the closest lane center.
    
    Args:
        lanes: the list of lanes to process
    
    Returns:
        center_intercept: the horizontal intercept of the center of the closest lane
        center_slope: the slope of the closest lane center
    """
    if not lanes or len(lanes) == 0:
        return None, None
    
    # Take the first (closest) lane
    lane = lanes[0]
    if len(lane) != 2:
        return None, None
    
    line1, line2 = lane
    
    # Extract coordinates from both lines
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]
    
    # Calculate center points at both ends
    center_x1 = (x1_1 + x1_2) / 2
    center_y1 = (y1_1 + y1_2) / 2
    center_x2 = (x2_1 + x2_2) / 2
    center_y2 = (y2_1 + y2_2) / 2
    
    # Calculate slope of the center line
    if abs(center_x2 - center_x1) < 1e-6:  # Nearly vertical center line
        center_slope = float('inf')
        center_intercept = center_x1
    else:
        center_slope = (center_y2 - center_y1) / (center_x2 - center_x1)
        # Calculate horizontal intercept (x-intercept): y = mx + b, so x = -b/m when y = 0
        # First find y-intercept: b = y - mx
        y_intercept = center_y1 - center_slope * center_x1
        # Then find x-intercept: x = -b/m
        if abs(center_slope) < 1e-6:  # Nearly horizontal line
            center_intercept = float('inf')
        else:
            center_intercept = -y_intercept / center_slope
    
    return center_intercept, center_slope

def recommend_direction(center, slope):
    """
    Recommends a direction based on the center and slope of the closest lane.
    
    Args:
        center: the center of the closest lane
        slope: the slope of the closest lane
    
    Returns:
        direction: the recommended direction ('left', 'right', or 'forward')
    """
    if center is None:
        return "forward"  # Default when no lane is detected
    
    # For this implementation, we'll use a simple approach based on the center position
    # We need to make assumptions about image dimensions since they're not provided
    # Assuming a typical image width, we'll use relative positioning
    
    # If center is negative, it's on the left side of the coordinate system
    # If center is positive and large, it's on the right side
    # We'll use reasonable thresholds
    
    if center < -100:  # Far left
        return "right"  # Need to steer right to get back to center
    elif center > 100:   # Far right  
        return "left"   # Need to steer left to get back to center
    else:  # In the middle range
        return "forward"