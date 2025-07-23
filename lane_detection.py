# lane_detection.py

import cv2
import numpy as np

def detect_lines(img, threshold1=20, threshold2=60, minLineLength=200, maxLineGap=25):
    """
    Detects lines in an image. Parameters are tuned to be less strict to ensure
    detection in low-contrast environments.
    """
    # Convert image to grayscale and apply a Gaussian blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using the Canny algorithm
    edges = cv2.Canny(blurred, threshold1, threshold2, apertureSize=3)
    
    # Detect lines using the Probabilistic Hough Line Transform
    # The Hough `threshold` is lowered to accept lines with fewer points
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,  # Lowered threshold to find more candidate lines
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    return lines

def average_lanes(img, lines):
    """
    Filters and averages detected lines into a stable left and right lane line.
    """
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Avoid division by zero
        if x1 == x2:
            continue
            
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Separate lines into left and right based on slope
        # A slope tolerance is added to ignore near-horizontal lines
        if slope < -0.3:  # Left lane
            left_fit.append((slope, intercept))
        elif slope > 0.3: # Right lane
            right_fit.append((slope, intercept))
            
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    
    def make_coordinates(image, line_parameters):
        """Helper function to convert slope and intercept into coordinates."""
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    averaged_lines = []
    if left_fit_average is not None:
        averaged_lines.append(make_coordinates(img, left_fit_average))
    if right_fit_average is not None:
        averaged_lines.append(make_coordinates(img, right_fit_average))
        
    return averaged_lines if averaged_lines else None

detect_lanes = average_lanes

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    """Draws the final, averaged lane lines on the image."""
    img_with_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img_with_lines, (x1, y1), (x2, y2), color, thickness)
    return img_with_lines

draw_lanes = draw_lines