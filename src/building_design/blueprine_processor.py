"""
Process the blueprint of the building
"""
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
from torch_geometric.data import Data

class BlueprintProcessor:
    """Process building blueprints to extract graph structure for BuildingGemNet"""
    
    def __init__(self, tesseract_path=None):
        """
        Initialize the blueprint processor
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # ... existing methods ...
    
    def _extract_room_features(self, img, x, y, w, h, dims):
        """
        Extract features for room nodes
        
        Args:
            img: Source image
            x, y, w, h: Bounding box coordinates
            dims: Extracted dimensions text
            
        Returns:
            Feature vector for room
        """
        # Parse dimensions from text (format like "10' x 12'")
        width, height = self._parse_dimensions(dims)
        
        # Calculate area and perimeter
        area = width * height if width and height else 0
        perimeter = 2 * (width + height) if width and height else 0
        
        # Extract room color/brightness (might indicate room type)
        roi = img[y:y+h, x:x+w]
        avg_color = np.mean(roi, axis=(0, 1)) if roi.size > 0 else np.zeros(3)
        brightness = np.mean(avg_color)
        
        # Check for windows (simplified - look for thin rectangles near walls)
        has_windows = self._detect_windows(img, x, y, w, h)
        
        # Create feature vector
        # [width, height, area, perimeter, shape_complexity, 
        #  brightness, has_windows, is_corner_room, occupancy_estimate, purpose_encoding]
        features = np.zeros(10)
        features[0] = width
        features[1] = height
        features[2] = area
        features[3] = perimeter
        features[4] = perimeter / (4 * np.sqrt(area)) if area > 0 else 1  # Shape complexity
        features[5] = brightness / 255.0  # Normalize brightness
        features[6] = 1.0 if has_windows else 0.0
        features[7] = self._is_corner_room(x, y, w, h, img.shape)
        features[8] = self._estimate_occupancy(area)
        features[9] = 0.5  # Default room purpose encoding
        
        return features
    
    def _extract_hvac_features(self, img, x, y, w, h):
        """
        Extract features for HVAC nodes
        
        Args:
            img: Source image
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Feature vector for HVAC unit
        """
        # Look for HVAC symbols near the text
        roi = self._expand_roi(img, x, y, w, h, padding=50)
        has_diffusers = self._detect_diffusers(roi)
        has_ducts = self._detect_ducts(roi)
        
        # Check for text indicating capacity
        nearby_text = self._get_nearby_text(x, y, w, h, radius=100)
        capacity_kw = self._extract_capacity_from_text(nearby_text)
        
        # Create feature vector
        # [capacity_kw, unit_size, centrality, has_diffusers, has_ducts,
        #  is_rooftop, age_estimation, efficiency_class]
        features = np.zeros(8)
        features[0] = capacity_kw if capacity_kw else 10.0  # Default 10kW if unknown
        features[1] = np.sqrt(w * h) / 100.0  # Normalized size
        features[2] = self._calculate_centrality(x + w/2, y + h/2, img.shape)
        features[3] = 1.0 if has_diffusers else 0.0
        features[4] = 1.0 if has_ducts else 0.0
        features[5] = 1.0 if y < img.shape[0] * 0.2 else 0.0  # Is rooftop (near top of image)
        features[6] = 0.5  # Default mid-age
        features[7] = 0.7  # Default good efficiency
        
        return features
    
    def _extract_electrical_features(self, img, x, y, w, h):
        """
        Extract features for electrical panel nodes
        
        Args:
            img: Source image
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Feature vector for electrical panel
        """
        # Look for electrical symbols near the text
        roi = self._expand_roi(img, x, y, w, h, padding=50)
        has_symbol = self._detect_electrical_symbols(roi)
        
        # Check for text indicating amperage
        nearby_text = self._get_nearby_text(x, y, w, h, radius=100)
        amperage = self._extract_amperage_from_text(nearby_text)
        
        # Check for nearby outlets/switches
        num_outlets = self._count_nearby_outlets(img, x, y, w, h)
        
        # Create feature vector
        # [amperage, panel_size, is_main_panel, num_circuits, 
        #  has_symbol, num_nearby_outlets]
        features = np.zeros(6)
        features[0] = amperage if amperage else 100.0  # Default 100A if unknown
        features[1] = np.sqrt(w * h) / 50.0  # Normalized size
        features[2] = 1.0 if "MAIN" in nearby_text.upper() else 0.0
        features[3] = 20.0  # Default number of circuits
        features[4] = 1.0 if has_symbol else 0.0
        features[5] = min(num_outlets / 10.0, 1.0)  # Normalize by 10 with cap at 1.0
        
        return features
    
    def _parse_dimensions(self, dims_text):
        """Parse dimensions from text"""
        if not dims_text:
            return None, None
        
        # Look for patterns like "10' x 12'" or "10ft x 12ft" or "3.5m x 4m"
        match = re.search(r'(\d+\.?\d*)[\'\"]?\s*[xX]\s*(\d+\.?\d*)[\'\"]?', dims_text)
        if match:
            return float(match.group(1)), float(match.group(2))
            
        return None, None
    
    def _detect_windows(self, img, x, y, w, h, padding=20):
        """Detect if room has windows (simplified)"""
        # Expand region to include walls
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi
        
        # Look for thin rectangular shapes that could be windows
        # This is a simplified approach - would be more sophisticated in practice
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = max(w, h) / (min(w, h) + 0.01)  # Avoid division by zero
            area = w * h
            
            # Windows typically have high aspect ratio and small-medium area
            if 3 < ratio < 10 and 100 < area < 1000:
                return True
                
        return False
    
    def _is_corner_room(self, x, y, w, h, img_shape):
        """Determine if room is likely a corner room"""
        # Check if room is near the edges of the blueprint
        near_left = x < img_shape[1] * 0.15
        near_right = (x + w) > img_shape[1] * 0.85
        near_top = y < img_shape[0] * 0.15
        near_bottom = (y + h) > img_shape[0] * 0.85
        
        # Corner room should be near at least two edges
        corner_score = near_left + near_right + near_top + near_bottom
        return 1.0 if corner_score >= 2 else 0.0
    
    def _estimate_occupancy(self, area):
        """Estimate room occupancy based on area"""
        # Rough estimate: 1 person per 10m² (or 100ft²)
        # Normalized to range [0, 1] with 1.0 being 10+ people
        return min(area / 1000.0, 1.0)
    
    def _expand_roi(self, img, x, y, w, h, padding):
        """Expand region of interest with padding"""
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        return img[y1:y2, x1:x2]
    
    def _detect_diffusers(self, roi):
        """Detect HVAC diffuser symbols in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi
        
        # Look for circular or square patterns with internal lines (simplified)
        # Would use more sophisticated template matching in practice
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20, 
            param1=50, param2=30, minRadius=10, maxRadius=30
        )
        
        return circles is not None and len(circles[0]) > 0
    
    def _detect_ducts(self, roi):
        """Detect HVAC duct symbols in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi
        
        # Look for parallel lines that could be ducts
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        return lines is not None and len(lines) >= 2
    
    def _get_nearby_text(self, x, y, w, h, radius):
        """Get text near a region (would use OCR results)"""
        # In a real implementation, this would search through OCR results
        # For this skeleton, we'll just return an empty string
        return ""
    
    def _extract_capacity_from_text(self, text):
        """Extract HVAC capacity from text (e.g., "10kW")"""
        match = re.search(r'(\d+\.?\d*)\s*[kK][wW]', text)
        return float(match.group(1)) if match else None
    
    def _calculate_centrality(self, x, y, img_shape):
        """Calculate how central a point is in the image"""
        center_x = img_shape[1] / 2
        center_y = img_shape[0] / 2
        
        # Distance from center, normalized by image dimensions
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt((center_x)**2 + (center_y)**2)
        
        # Convert to centrality (1.0 = center, 0.0 = edge)
        return 1.0 - (dist / max_dist)
    
    def _detect_electrical_symbols(self, roi):
        """Detect electrical symbols in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi
        
        # Look for electrical symbols (simplified)
        # Would use template matching in practice
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Some electrical symbols are square with internal details
            if 0.8 < aspect_ratio < 1.2 and 100 < w*h < 400:
                return True
                
        return False
    
    def _extract_amperage_from_text(self, text):
        """Extract electrical panel amperage from text (e.g., "200A")"""
        match = re.search(r'(\d+)\s*[aA]', text)
        return float(match.group(1)) if match else None
    
    def _count_nearby_outlets(self, img, x, y, w, h, radius=100):
        """Count electrical outlets near panel"""
        # This would use symbol detection in the full implementation
        # For the skeleton, we'll return a random count
        return np.random.randint(0, 10)