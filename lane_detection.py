import cv2
import numpy as np
import matplotlib.pyplot as plt

# lane_detection.py functions
def detect_lines(img, threshold1=20, threshold2=60, apertureSize=3, minLineLength=200, maxLineGap=25):
    """
    Detects lines in an image using the Canny edge detector and Hough Transform.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    
    
    # Detect edges using the Canny algorithm
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize)
    
    # Create a region of interest (ROI) mask - focus on lower portion of image
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width//4, height//2), 
                            (3*width//4, height//2), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using the Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    return lines

def draw_lines(img, lines, color=(0, 255, 0)):
    """
    Draws a list of lines onto an image.
    """
    if img is None:
        return None
        
    # Create a copy to draw on
    img_with_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), color, 2)
    return img_with_lines

def get_slopes_intercepts(lines):
    """
    Calculates the slope and y-intercept for each line.
    Note: Vertical lines are ignored to prevent division by zero.
    """
    slopes = []
    intercepts = []
    if lines is None:
        return slopes, intercepts
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1e-6:  # Avoid division by zero for vertical lines
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        slopes.append(slope)
        intercepts.append(intercept)
        
    return slopes, intercepts

def detect_lanes(lines):
    """
    Groups lines into pairs that form a lane.
    A lane is defined as two lines that are nearly parallel and appropriately spaced.
    """
    lanes = []
    if lines is None or len(lines) < 2:
        return lanes

    # Filter lines by slope to focus on lane-like lines
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1e-6:  # Skip vertical lines
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Keep lines with reasonable slopes (not too steep or too flat)
        if 0.1 <= abs(slope) <= 10:
            filtered_lines.append(line)
    
    if len(filtered_lines) < 2:
        return lanes

    # Parameters to define a lane
    SLOPE_TOLERANCE = 0.3  # Max difference in slope to be considered parallel
    MIN_DISTANCE = 30      # Min horizontal distance between lines
    MAX_DISTANCE = 200     # Max horizontal distance between lines

    used_indices = set()
    
    for i in range(len(filtered_lines)):
        if i in used_indices:
            continue
            
        line1 = filtered_lines[i][0]
        x1_1, y1_1, x2_1, y2_1 = line1
        
        # Calculate slope and midpoint of the first line
        slope1 = (y2_1 - y1_1) / (x2_1 - x1_1)
        mid_y1 = (y1_1 + y2_1) / 2
        mid_x1 = (x1_1 + x2_1) / 2

        best_match_idx = -1
        best_score = float('inf')

        for j in range(i + 1, len(filtered_lines)):
            if j in used_indices:
                continue

            line2 = filtered_lines[j][0]
            x1_2, y1_2, x2_2, y2_2 = line2

            # Calculate slope and midpoint of the second line
            slope2 = (y2_2 - y1_2) / (x2_2 - x1_2)
            mid_y2 = (y1_2 + y2_2) / 2
            mid_x2 = (x1_2 + x2_2) / 2

            # Check if slopes are similar (parallel lines)
            slope_diff = abs(slope1 - slope2)
            if slope_diff > SLOPE_TOLERANCE:
                continue
                
            # Calculate horizontal distance between line midpoints
            horizontal_dist = abs(mid_x1 - mid_x2)
            
            # Check if lines are reasonably spaced
            if MIN_DISTANCE <= horizontal_dist <= MAX_DISTANCE:
                # Score based on slope similarity and reasonable spacing
                score = slope_diff + (abs(horizontal_dist - 100) / 1000)  # Prefer ~100px spacing
                
                if score < best_score:
                    best_score = score
                    best_match_idx = j
        
        if best_match_idx != -1:
            lanes.append([filtered_lines[i], filtered_lines[best_match_idx]])
            used_indices.add(i)
            used_indices.add(best_match_idx)
            
    return lanes

def draw_lanes(img, lanes):
    """
    Draws detected lanes on an image, each with a different color.
    """
    if img is None:
        return None
        
    img_with_lanes = img.copy()
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    if lanes is not None:
        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            for line in lane:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_with_lanes, (x1, y1), (x2, y2), color, 3)
    return img_with_lanes