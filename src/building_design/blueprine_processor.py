"""
Process the blueprint of the building
"""
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
    
    def process_blueprint(self, image_path):
        """
        Process a blueprint image to extract building graph
        
        Args:
            image_path: Path to blueprint image
            
        Returns:
            torch_geometric.data.Data object with graph structure
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Extract text with OCR
        text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        
        # Extract nodes (rooms, HVAC, electrical panels)
        rooms, hvac_units, electrical_panels = self._extract_nodes(img, text_data)
        
        # Extract edges (connections between nodes)
        edges = self._extract_edges(img, rooms, hvac_units, electrical_panels)
        
        # Create graph data
        return self._create_graph_data(rooms, hvac_units, electrical_panels, edges)
    
    def _extract_nodes(self, img, text_data):
        """Extract nodes from blueprint based on text labels and shapes"""
        # Example implementation - in practice would use more sophisticated CV techniques
        rooms = []
        hvac_units = []
        electrical_panels = []
        
        # Extract room information from text
        for i, text in enumerate(text_data['text']):
            if text.strip():
                x = text_data['left'][i]
                y = text_data['top'][i]
                w = text_data['width'][i]
                h = text_data['height'][i]
                
                # Simple heuristic to categorize text
                if "ROOM" in text.upper() or "OFFICE" in text.upper() or "HALL" in text.upper():
                    # Get room dimensions by finding nearby numerical text
                    dims = self._find_nearby_dimensions(text_data, i)
                    rooms.append({
                        'id': len(rooms),
                        'type': 'room',
                        'label': text.strip(),
                        'position': (x + w//2, y + h//2),
                        'dimensions': dims,
                        'features': self._extract_room_features(img, x, y, w, h, dims)
                    })
                
                elif "HVAC" in text.upper() or "AC" in text.upper() or "HEATING" in text.upper():
                    hvac_units.append({
                        'id': len(hvac_units),
                        'type': 'hvac',
                        'label': text.strip(),
                        'position': (x + w//2, y + h//2),
                        'features': self._extract_hvac_features(img, x, y, w, h)
                    })
                
                elif "PANEL" in text.upper() or "ELECTRIC" in text.upper():
                    electrical_panels.append({
                        'id': len(electrical_panels),
                        'type': 'electrical',
                        'label': text.strip(),
                        'position': (x + w//2, y + h//2),
                        'features': self._extract_electrical_features(img, x, y, w, h)
                    })
        
        return rooms, hvac_units, electrical_panels
    
    def _extract_edges(self, img, rooms, hvac_units, electrical_panels):
        """Extract edges connecting nodes in the blueprint"""
        edges = []
        
        # Example: Use line detection to find connections
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_img = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        all_nodes = rooms + hvac_units + electrical_panels
        
        # Connect nodes based on detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Find nodes close to line endpoints
                node1 = self._find_closest_node(all_nodes, (x1, y1))
                node2 = self._find_closest_node(all_nodes, (x2, y2))
                
                if node1 is not None and node2 is not None and node1 != node2:
                    # Determine edge type based on node types
                    edge_type = self._determine_edge_type(node1, node2)
                    
                    if edge_type:
                        edges.append({
                            'source': node1['id'],
                            'target': node2['id'],
                            'type': edge_type,
                            'features': self._extract_edge_features(img, node1, node2, edge_type)
                        })
        
        return edges
    
    def _create_graph_data(self, rooms, hvac_units, electrical_panels, edges):
        """Create torch_geometric.data.Data from extracted nodes and edges"""
        # Combine all nodes
        all_nodes = rooms + hvac_units + electrical_panels
        
        # Create node type tensor
        node_types = torch.tensor([
            0 if node['type'] == 'room' else 
            1 if node['type'] == 'hvac' else 
            2 for node in all_nodes
        ], dtype=torch.long)
        
        # Create node features
        node_features = {
            'room': torch.tensor([node['features'] for node in rooms if node['type'] == 'room'], dtype=torch.float),
            'hvac': torch.tensor([node['features'] for node in all_nodes if node['type'] == 'hvac'], dtype=torch.float),
            'electrical': torch.tensor([node['features'] for node in all_nodes if node['type'] == 'electrical'], dtype=torch.float)
        }
        
        # Create edge index tensor
        edge_index = torch.tensor([[edge['source'], edge['target']] for edge in edges], dtype=torch.long).t()
        
        # Create edge type tensor
        edge_types = torch.tensor([edge['type'] for edge in edges], dtype=torch.long)
        
        # Create edge features
        edge_features = {
            'structural': torch.tensor([edge['features'] for edge in edges if edge['type'] == 0], dtype=torch.float),
            'pipes': torch.tensor([edge['features'] for edge in edges if edge['type'] == 1], dtype=torch.float),
            'electrical': torch.tensor([edge['features'] for edge in edges if edge['type'] == 2], dtype=torch.float),
            'airflow': torch.tensor([edge['features'] for edge in edges if edge['type'] == 3], dtype=torch.float)
        }
        
        # Create edge values for basis functions
        edge_values = torch.tensor([self._get_edge_value(edge) for edge in edges], dtype=torch.float)
        
        # Create edge capacities for flow basis
        edge_capacities = torch.tensor([self._get_edge_capacity(edge) for edge in edges], dtype=torch.float)
        
        # Create Data object
        data = Data(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_features=edge_features,
            edge_values=edge_values,
            edge_capacities=edge_capacities,
            batch=torch.zeros(len(all_nodes), dtype=torch.long)  # Single building
        )
        
        return data
    
    # Helper methods - these