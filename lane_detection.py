import cv2
import numpy as np

def detect_lines(img, threshold1=50, threshold2=150, apertureSize=3, minLineLength=100, maxLineGap=10):
    """
    Detects line segments in an image using the Hough Line Transform.
    
    Args:
        img: the image to process
        threshold1: the first threshold for the Canny edge detector (default: 50)
        threshold2: the second threshold for the Canny edge detector (default: 150)
        apertureSize: the aperture size for the Sobel operator (default: 3)
        minLineLength: the minimum length of a line (default: 100)
        maxLineGap: the maximum gap between two points to be considered in the same line (default: 10)
    
    Returns:
        lines: list of detected lines
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges with the Canny algorithm
    edges = cv2.Canny(blurred, threshold1, threshold2, apertureSize=apertureSize)
    
    # Use the Probabilistic Hough Transform to find line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    
    return lines

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    """
    Draws a list of line segments onto an image.
    
    Args:
        img: the image to process
        lines: the list of lines to draw
        color: the color of the lines (default: (0, 255, 0))
        thickness: thickness of the lines (default: 5)
    
    Returns:
        img_with_lines: image with lines drawn on it
    """
    if img is None:
        return None
        
    img_with_lines = img.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return img_with_lines

def get_slopes_intercepts(lines):
    """
    Calculates the slope and horizontal intercepts for each line segment.
    
    Args:
        lines: the list of lines to process
    
    Returns:
        slopes: the list of slopes
        intercepts: the list of horizontal intercepts
    """
    slopes = []
    intercepts = []
    
    if lines is None:
        return slopes, intercepts
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Avoid division by zero for vertical lines
        if abs(x2 - x1) < 1e-6:
            slopes.append(float('inf'))
            intercepts.append(x1)
        else:
            slope = (y2 - y1) / (x2 - x1)
            # Calculate horizontal intercept (x-intercept): y = mx + b, so x = -b/m when y = 0
            # First find y-intercept: b = y - mx
            y_intercept = y1 - slope * x1
            # Then find x-intercept: x = -b/m
            if abs(slope) < 1e-6:  # Nearly horizontal line
                x_intercept = float('inf')
            else:
                x_intercept = -y_intercept / slope
            
            slopes.append(slope)
            intercepts.append(x_intercept)
        
    return slopes, intercepts

def detect_lanes(lines):
    """
    Detects lanes from a list of lines by finding pairs that could form lanes.
    
    Args:
        lines: the list of lines to process
    
    Returns:
        lanes: the list of lanes (each lane is a list of two lines)
    """
    if lines is None or len(lines) < 2:
        return []
    
    # Get slopes and intercepts
    slopes, intercepts = get_slopes_intercepts(lines)
    
    lanes = []
    used_indices = set()
    
    # Check each pair of lines to see if they form a lane
    for i in range(len(lines)):
        if i in used_indices:
            continue
            
        for j in range(i + 1, len(lines)):
            if j in used_indices:
                continue
            
            # Check if this pair could be a lane
            slope1, slope2 = slopes[i], slopes[j]
            
            # Skip if either line is vertical
            if slope1 == float('inf') or slope2 == float('inf'):
                continue
            
            # Check if slopes have opposite signs (one positive, one negative)
            # and are reasonably similar in magnitude
            if (slope1 * slope2 < 0 and  # Opposite signs
                0.1 <= abs(slope1) <= 5.0 and  # Reasonable slope range
                0.1 <= abs(slope2) <= 5.0 and  # Reasonable slope range
                abs(abs(slope1) - abs(slope2)) < 2.0):  # Similar magnitudes
                
                # This pair forms a potential lane
                lanes.append([lines[i], lines[j]])
                used_indices.add(i)
                used_indices.add(j)
                break  # Move to next line
    
    return lanes

def draw_lanes(img, lanes, thickness=8):
    """
    Draws lanes onto an image with different colors for each lane.
    
    Args:
        img: the image to process
        lanes: the list of lanes to draw
        thickness: thickness of the lane lines (default: 8)
    
    Returns:
        img_with_lanes: image with lanes drawn on it
    """
    if img is None:
        return None
        
    img_with_lanes = img.copy()
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    if lanes:
        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            for line in lane:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_with_lanes, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
    return img_with_lanes