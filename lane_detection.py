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
        threshold=50,  # Reduced threshold to catch more lines
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    return lines

def remove_duplicate_lines(lines, slope_tolerance=0.08, position_tolerance=50):
    """
    Removes duplicate lines based on slope and position similarity.
    Uses a custom clustering approach without external dependencies.
    """
    if lines is None or len(lines) == 0:
        return lines
    
    # Extract line features
    line_data = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        # Skip very short or vertical lines
        if abs(x2 - x1) < 5:
            continue
            
        # Calculate line properties
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Only keep lines with reasonable slopes for lane detection
        if 0.1 <= abs(slope) <= 5.0:
            line_data.append({
                'index': i,
                'line': line,
                'slope': slope,
                'intercept': intercept,
                'mid_x': mid_x,
                'mid_y': mid_y,
                'length': length
            })
    
    if len(line_data) == 0:
        return None
    
    # Custom clustering: group similar lines
    clusters = []
    used_indices = set()
    
    for i, line1 in enumerate(line_data):
        if i in used_indices:
            continue
            
        # Start a new cluster
        cluster = [line1]
        cluster_indices = {i}
        
        # Find all similar lines
        for j, line2 in enumerate(line_data[i+1:], i+1):
            if j in used_indices:
                continue
                
            # Check similarity criteria
            slope_diff = abs(line1['slope'] - line2['slope'])
            position_diff = abs(line1['mid_x'] - line2['mid_x'])
            
            if slope_diff < slope_tolerance and position_diff < position_tolerance:
                cluster.append(line2)
                cluster_indices.add(j)
        
        # Mark all lines in this cluster as used
        used_indices.update(cluster_indices)
        clusters.append(cluster)
    
    # Keep the longest line from each cluster
    unique_lines = []
    for cluster in clusters:
        # Find the longest line in the cluster
        longest_line = max(cluster, key=lambda x: x['length'])
        unique_lines.append(longest_line['line'])
    
    return unique_lines if unique_lines else None

def classify_lines_by_position_and_slope(lines, img_width):
    """
    Classifies lines as left or right based on their position AND slope.
    More robust than using slope alone.
    """
    if lines is None:
        return [], []
    
    left_lines = []
    right_lines = []
    center_x = img_width / 2
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line properties
        if abs(x2 - x1) < 1e-6:  # Skip vertical lines
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        
        # Classify based on position AND slope
        if mid_x < center_x:  # Left side of image
            if slope < -0.2:  # Negative slope (as expected for left lane)
                left_lines.append(line)
        else:  # Right side of image
            if slope > 0.2:  # Positive slope (as expected for right lane)
                right_lines.append(line)
    
    return left_lines, right_lines

def merge_similar_lines_in_group(lines, merge_threshold=0.15):
    """
    Merges lines that are very similar within a group (left or right).
    """
    if not lines or len(lines) <= 1:
        return lines
    
    merged_lines = []
    used_indices = set()
    
    for i, line1 in enumerate(lines):
        if i in used_indices:
            continue
            
        x1_1, y1_1, x2_1, y2_1 = line1[0]
        slope1 = (y2_1 - y1_1) / (x2_1 - x1_1) if abs(x2_1 - x1_1) > 1e-6 else float('inf')
        
        # Find all similar lines
        similar_lines = [line1]
        similar_indices = [i]
        
        for j, line2 in enumerate(lines[i+1:], i+1):
            if j in used_indices:
                continue
                
            x1_2, y1_2, x2_2, y2_2 = line2[0]
            slope2 = (y2_2 - y1_2) / (x2_2 - x1_2) if abs(x2_2 - x1_2) > 1e-6 else float('inf')
            
            # Check if lines are similar
            if abs(slope1 - slope2) < merge_threshold:
                similar_lines.append(line2)
                similar_indices.append(j)
        
        # Mark all similar lines as used
        for idx in similar_indices:
            used_indices.add(idx)
        
        # Merge similar lines if there are multiple
        if len(similar_lines) > 1:
            # Collect all points from similar lines
            all_x = []
            all_y = []
            for line in similar_lines:
                x1, y1, x2, y2 = line[0]
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
            
            # Fit a line through all points using least squares
            if len(set(all_x)) > 1:  # Avoid vertical lines
                coeffs = np.polyfit(all_x, all_y, 1)
                slope, intercept = coeffs
                
                # Create merged line spanning the range of all similar lines
                min_x = min(all_x)
                max_x = max(all_x)
                y1_new = slope * min_x + intercept
                y2_new = slope * max_x + intercept
                
                merged_line = np.array([[int(min_x), int(y1_new), int(max_x), int(y2_new)]])
                merged_lines.append(merged_line)
            else:
                # Keep the longest original line if can't merge
                longest = max(similar_lines, key=lambda l: np.sqrt((l[0][2] - l[0][0])**2 + (l[0][3] - l[0][1])**2))
                merged_lines.append(longest)
        else:
            merged_lines.append(line1)
    
    return merged_lines

def extend_line_to_full_height(line, img_height, img_width):
    """
    Extends a line to span the full height of the image.
    """
    x1, y1, x2, y2 = line[0] if isinstance(line, np.ndarray) and len(line.shape) > 1 else line
    
    if abs(x2 - x1) < 1e-6:  # Vertical line
        return [int(x1), 0, int(x1), img_height-1]
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Calculate x coordinates at top and bottom of image
    y_top = 0
    y_bottom = img_height - 1
    x_top = (y_top - intercept) / slope if abs(slope) > 1e-6 else x1
    x_bottom = (y_bottom - intercept) / slope if abs(slope) > 1e-6 else x1
    
    # Clamp to image bounds
    x_top = max(0, min(img_width-1, x_top))
    x_bottom = max(0, min(img_width-1, x_bottom))
    
    return [int(x_top), y_top, int(x_bottom), y_bottom]

def detect_lanes(img, lines):
    """
    Improved lane detection with better duplicate handling and left/right classification.
    """
    if lines is None:
        return []
    
    img_height, img_width = img.shape[:2]
    
    # Step 1: Remove duplicate lines
    print(f"Original lines: {len(lines)}")
    unique_lines = remove_duplicate_lines(lines)
    
    if unique_lines is None or len(unique_lines) == 0:
        print("No valid lines after duplicate removal")
        return []
    
    print(f"After duplicate removal: {len(unique_lines)}")
    
    # Step 2: Classify lines by position and slope
    left_lines, right_lines = classify_lines_by_position_and_slope(unique_lines, img_width)
    print(f"Left lines: {len(left_lines)}, Right lines: {len(right_lines)}")
    
    # Step 3: Merge very similar lines within each side
    left_lines = merge_similar_lines_in_group(left_lines)
    right_lines = merge_similar_lines_in_group(right_lines)
    print(f"After merging - Left: {len(left_lines)}, Right: {len(right_lines)}")
    
    # Step 4: Create lanes by pairing left and right lines
    lanes = []
    
    # If we have both left and right lines, create lanes
    if left_lines and right_lines:
        # Select the best left and right line (longest ones)
        def get_line_length(line):
            coords = line[0] if isinstance(line, np.ndarray) and len(line.shape) > 1 else line
            x1, y1, x2, y2 = coords
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        best_left = max(left_lines, key=get_line_length)
        best_right = max(right_lines, key=get_line_length)
        
        # Extend lines to full image height
        extended_left = extend_line_to_full_height(best_left, img_height, img_width)
        extended_right = extend_line_to_full_height(best_right, img_height, img_width)
        
        lanes.append([extended_left, extended_right])
    
    return lanes

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    """
    Draws a list of line segments onto an image.
    """
    if img is None:
        return None
        
    img_with_lines = img.copy()
    if lines is not None:
        for line in lines:
            if isinstance(line, np.ndarray) and len(line.shape) > 1:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            cv2.line(img_with_lines, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img_with_lines

def draw_lanes(img, lanes, thickness=8):
    """
    Draws the final lanes onto an image with different colors.
    """
    if img is None:
        return None
        
    img_with_lanes = img.copy()
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    if lanes:
        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            for line in lane:
                if isinstance(line, (list, tuple)) and len(line) == 4:
                    x1, y1, x2, y2 = line
                else:
                    x1, y1, x2, y2 = line[0] if hasattr(line, '__len__') and len(line) > 0 else line
                cv2.line(img_with_lanes, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
    return img_with_lanes

def get_slopes_intercepts(lines):
    """
    Calculates the slope and y-intercept for each line segment.
    """
    slopes = []
    intercepts = []
    
    if lines is None:
        return slopes, intercepts
        
    for line in lines:
        if isinstance(line, np.ndarray) and len(line.shape) > 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line
        
        # Avoid division by zero for vertical lines
        if abs(x2 - x1) < 1e-6:
            slopes.append(float('inf'))
            intercepts.append(x1)
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)
        
    return slopes, intercepts