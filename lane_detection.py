import cv2
import numpy as np
import random

def detect_lines(img, threshold1=20, threshold2=60, apertureSize=3, minLineLength=200, maxLineGap=25):
    """
    Detects line segments in an image using the Hough Line Transform.
    """
    # 1. Convert to grayscale and apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Detect edges with the Canny algorithm on the blurred image
    edges = cv2.Canny(blurred, threshold1, threshold2, apertureSize=apertureSize)
    
    # 3. Use the Probabilistic Hough Transform to find line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,  # A good starting threshold
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    return lines

def draw_lines(img, lines, color=(0, 255, 0), thickness=20):
    """
    Draws a list of line segments onto an image.
    """
    img_with_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw each line with the specified color and thickness
            cv2.line(img_with_lines, (x1, y1), (x2, y2), color, thickness)
    return img_with_lines

def get_slopes_intercepts(lines):
    """
    Calculates the slope and y-intercept for each line segment.
    This function now correctly processes all lines without trying to filter them.
    """
    slopes = []
    intercepts = []
    
    if lines is None:
        return slopes, intercepts
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Avoid division by zero for vertical lines
        if x1 == x2:
            continue
            
        # Calculate slope and y-intercept (y = mx + b)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        slopes.append(slope)
        intercepts.append(intercept)
        
    return slopes, intercepts

def detect_lanes(img, lines):
    """
    Filters and averages line segments into a single representative lane.
    This is the primary function for handling multiple detections of the same lane line.
    """
    left_fit = []
    right_fit = []
    
    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: continue
        
        # Use np.polyfit to robustly find the slope and intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        
        # Group lines into "left" and "right" based on their slope
        if slope < -0.3:
            left_fit.append((slope, intercept))
        elif slope > 0.3:
            right_fit.append((slope, intercept))
            
    # Average the parameters for all lines in each group
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    
    def make_coordinates(image, line_parameters):
        """Helper to convert averaged slope/intercept back into screen coordinates."""
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    # If both a left and a right line are found, they form a lane
    if left_fit_average is not None and right_fit_average is not None:
        left_lane = make_coordinates(img, left_fit_average)
        right_lane = make_coordinates(img, right_fit_average)
        return [[left_lane, right_lane]]
        
    return []

def draw_lanes(img, lanes, thickness=10):
    """
    Draws the final, averaged lanes onto an image.
    """
    img_with_lanes = img.copy()
    if not lanes:
        return img_with_lanes
        
    for lane in lanes:
        # Each lane is a pair of lines. Draw them in a random color.
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for x1, y1, x2, y2 in lane:
            cv2.line(img_with_lanes, (x1, y1), (x2, y2), color, thickness)
            
    return img_with_lanes