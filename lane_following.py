import cv2
import numpy as np

def get_lane_center(lanes, img_height, img_width):
    """
    Finds the center intercept and slope of the detected lane.
    Returns the x-coordinate where the lane center intersects the bottom of the image.
    """
    if not lanes or len(lanes) == 0:
        return None, None
    
    # Take the first (best) lane
    lane = lanes[0]
    if len(lane) != 2:
        return None, None
    
    left_line, right_line = lane
    
    # Extract coordinates
    if isinstance(left_line, (list, tuple)):
        left_x1, left_y1, left_x2, left_y2 = left_line
    else:
        left_x1, left_y1, left_x2, left_y2 = left_line[0]
        
    if isinstance(right_line, (list, tuple)):
        right_x1, right_y1, right_x2, right_y2 = right_line
    else:
        right_x1, right_y1, right_x2, right_y2 = right_line[0]
    
    # Calculate center points at top and bottom of the lane
    center_x1 = (left_x1 + right_x1) / 2
    center_y1 = (left_y1 + right_y1) / 2
    center_x2 = (left_x2 + right_x2) / 2
    center_y2 = (left_y2 + right_y2) / 2
    
    # Calculate slope of the lane center line
    if abs(center_x2 - center_x1) < 1e-6:  # Nearly vertical center line
        center_slope = float('inf')
        center_intercept = center_x1
    else:
        center_slope = (center_y2 - center_y1) / (center_x2 - center_x1)
        # Calculate y-intercept: y = mx + b, so b = y - mx
        y_intercept = center_y1 - center_slope * center_x1
        
        # Calculate x-coordinate where center line intersects bottom of image
        # At y = img_height - 1: x = (y - b) / m
        if abs(center_slope) < 1e-6:  # Nearly horizontal line
            center_intercept = img_width / 2  # Default to image center
        else:
            center_intercept = (img_height - 1 - y_intercept) / center_slope
    
    # Ensure the intercept is within image bounds
    center_intercept = max(0, min(img_width - 1, center_intercept))
    
    return center_intercept, center_slope

def recommend_direction(center_intercept, img_width, tolerance_factor=0.08):
    """
    Recommends a steering direction based on the lane's center position.
    
    Args:
        center_intercept: X-coordinate where lane center meets bottom of image
        img_width: Width of the image in pixels
        tolerance_factor: Fraction of image width to use as tolerance (default 8%)
    
    Returns:
        String recommendation for steering direction
    """
    if center_intercept is None:
        return "No lane detected"
    
    image_center_x = img_width / 2
    tolerance = img_width * tolerance_factor
    
    offset = center_intercept - image_center_x
    
    if offset < -tolerance:
        return "Steer Left ⬅️"
    elif offset > tolerance:
        return "Steer Right ➡️"
    else:
        return "Go Forward ⬆️"

def get_steering_angle(center_intercept, img_width, max_angle=30):
    """
    Calculates a steering angle based on the lane center offset.
    
    Args:
        center_intercept: X-coordinate where lane center meets bottom of image
        img_width: Width of the image in pixels
        max_angle: Maximum steering angle in degrees
    
    Returns:
        Steering angle in degrees (negative = left, positive = right)
    """
    if center_intercept is None:
        return 0
    
    image_center_x = img_width / 2
    offset = center_intercept - image_center_x
    
    # Normalize offset to [-1, 1] range
    normalized_offset = offset / (img_width / 2)
    
    # Calculate steering angle proportional to offset
    steering_angle = normalized_offset * max_angle
    
    # Clamp to maximum angle
    steering_angle = max(-max_angle, min(max_angle, steering_angle))
    
    return steering_angle

def analyze_lane_curvature(lanes, img_height):
    """
    Analyzes the curvature of the detected lane.
    
    Args:
        lanes: List of detected lanes
        img_height: Height of the image
    
    Returns:
        Dictionary with curvature analysis
    """
    if not lanes or len(lanes) == 0:
        return {"curvature": "unknown", "curve_direction": "unknown", "curve_strength": 0}
    
    lane = lanes[0]
    if len(lane) != 2:
        return {"curvature": "unknown", "curve_direction": "unknown", "curve_strength": 0}
    
    left_line, right_line = lane
    
    # Extract coordinates
    if isinstance(left_line, (list, tuple)):
        left_x1, left_y1, left_x2, left_y2 = left_line
    else:
        left_x1, left_y1, left_x2, left_y2 = left_line[0]
        
    if isinstance(right_line, (list, tuple)):
        right_x1, right_y1, right_x2, right_y2 = right_line
    else:
        right_x1, right_y1, right_x2, right_y2 = right_line[0]
    
    # Calculate slopes
    left_slope = (left_y2 - left_y1) / (left_x2 - left_x1) if abs(left_x2 - left_x1) > 1e-6 else float('inf')
    right_slope = (right_y2 - right_y1) / (right_x2 - right_x1) if abs(right_x2 - right_x1) > 1e-6 else float('inf')
    
    # Calculate lane width at top and bottom
    width_top = abs(right_x1 - left_x1)
    width_bottom = abs(right_x2 - left_x2)
    
    # Analyze curvature based on slope difference and width change
    slope_diff = right_slope + left_slope  # Should be ~0 for straight lanes
    width_change = abs(width_top - width_bottom)
    
    # Determine curve characteristics
    if abs(slope_diff) < 0.1 and width_change < 20:
        curvature = "straight"
        curve_direction = "none"
        curve_strength = 0
    elif slope_diff > 0.1:
        curvature = "curved"
        curve_direction = "right"
        curve_strength = min(abs(slope_diff), 1.0)
    elif slope_diff < -0.1:
        curvature = "curved"
        curve_direction = "left"
        curve_strength = min(abs(slope_diff), 1.0)
    else:
        curvature = "slight_curve"
        curve_direction = "right" if slope_diff > 0 else "left"
        curve_strength = min(abs(slope_diff), 0.5)
    
    return {
        "curvature": curvature,
        "curve_direction": curve_direction,
        "curve_strength": curve_strength,
        "slope_difference": slope_diff,
        "width_change": width_change
    }

def get_lane_width(lanes, img_height):
    """
    Calculates the width of the detected lane at the bottom of the image.
    
    Args:
        lanes: List of detected lanes
        img_height: Height of the image
    
    Returns:
        Lane width in pixels, or None if no lane detected
    """
    if not lanes or len(lanes) == 0:
        return None
    
    lane = lanes[0]
    if len(lane) != 2:
        return None
    
    left_line, right_line = lane
    
    # Extract bottom coordinates (assuming lines extend to bottom of image)
    if isinstance(left_line, (list, tuple)):
        left_x_bottom = left_line[2]  # x2 coordinate
    else:
        left_x_bottom = left_line[0][2]
        
    if isinstance(right_line, (list, tuple)):
        right_x_bottom = right_line[2]  # x2 coordinate
    else:
        right_x_bottom = right_line[0][2]
    
    lane_width = abs(right_x_bottom - left_x_bottom)
    return lane_width

def get_detailed_lane_info(lanes, img_height, img_width):
    """
    Provides comprehensive information about the detected lane.
    
    Args:
        lanes: List of detected lanes
        img_height: Height of the image
        img_width: Width of the image
    
    Returns:
        Dictionary with detailed lane information
    """
    center_x, center_slope = get_lane_center(lanes, img_height, img_width)
    direction = recommend_direction(center_x, img_width)
    steering_angle = get_steering_angle(center_x, img_width)
    curvature_info = analyze_lane_curvature(lanes, img_height)
    lane_width = get_lane_width(lanes, img_height)
    
    info = {
        "center_x": center_x,
        "center_slope": center_slope,
        "direction_recommendation": direction,
        "steering_angle": steering_angle,
        "lane_width": lane_width,
        "offset_from_center": center_x - img_width/2 if center_x is not None else None,
        "offset_percentage": ((center_x - img_width/2) / (img_width/2)) * 100 if center_x is not None else None,
    }
    
    # Add curvature information
    info.update(curvature_info)
    
    return info