"""
Read the data for various inputs
"""
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import cv2
import ezdxf  # For CAD files
import fitz  # PyMuPDF for PDF files
import xml.etree.ElementTree as ET  # For Visio XML and SVG

class BlueprintReader:
    """Base class for reading building blueprint files"""
    
    def __init__(self, config=None):
        """Initialize with optional configuration"""
        self.config = config or {}
        
    def read(self, file_path):
        """Read file and return structured data"""
        raise NotImplementedError("Subclasses must implement this method")


class PDFBlueprintReader(BlueprintReader):
    """Reader for PDF blueprint files"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize OCR if needed
        if self.config.get('use_ocr', True):
            self.tesseract_path = self.config.get('tesseract_path')
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
    
    def read(self, file_path):
        """Read PDF blueprint and extract content"""
        # Open the PDF
        doc = fitz.open(file_path)
        result = {
            'pages': [],
            'text_blocks': [],
            'images': [],
            'vector_graphics': []
        }
        
        # Process each page
        for page_idx, page in enumerate(doc):
            # Extract text
            text_blocks = []
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # Text block
                    text_blocks.append({
                        'text': block.get("text", ""),
                        'bbox': block.get("bbox", []),
                        'page': page_idx
                    })
            
            # Extract images if present
            page_images = []
            for img_idx, img in enumerate(page.get_images(full=True)):
                # Placeholder for image extraction
                page_images.append({
                    'index': img_idx,
                    'page': page_idx,
                    # Additional image processing would go here
                })
            
            # Extract vector graphics (simplified)
            vector_paths = []
            # Would extract paths, lines, etc. here
                
            # Store page data
            result['pages'].append({
                'index': page_idx,
                'width': page.rect.width,
                'height': page.rect.height
            })
            result['text_blocks'].extend(text_blocks)
            result['images'].extend(page_images)
            result['vector_graphics'].extend(vector_paths)
        
        return result


class CADBlueprintReader(BlueprintReader):
    """Reader for CAD/DXF blueprint files"""
    
    def read(self, file_path):
        """Read CAD/DXF blueprint and extract content"""
        # Open DXF file
        doc = ezdxf.readfile(file_path)
        model_space = doc.modelspace()
        
        result = {
            'entities': [],
            'layers': [],
            'blocks': []
        }
        
        # Extract layers
        for layer in doc.layers:
            result['layers'].append({
                'name': layer.dxf.name,
                'color': layer.dxf.color,
                'linetype': layer.dxf.linetype
            })
        
        # Extract entities
        for entity in model_space:
            entity_data = {
                'type': entity.dxfattribs()['dxftype'],
                'layer': entity.dxf.layer,
                'handle': entity.dxf.handle
            }
            
            # Entity-specific properties
            if entity.dxftype() == 'LINE':
                entity_data.update({
                    'start': (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z),
                    'end': (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
                })
            elif entity.dxftype() == 'CIRCLE':
                entity_data.update({
                    'center': (entity.dxf.center.x, entity.dxf.center.y, entity.dxf.center.z),
                    'radius': entity.dxf.radius
                })
            elif entity.dxftype() == 'TEXT':
                entity_data.update({
                    'text': entity.dxf.text,
                    'position': (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                })
            
            result['entities'].append(entity_data)
        
        return result


class VisioXMLReader(BlueprintReader):
    """Reader for Visio XML/VSDX files"""
    
    def read(self, file_path):
        """Read Visio XML and extract content"""
        # For VSDX (which is a ZIP file), extract and parse
        if file_path.endswith('.vsdx'):
            # Would use zipfile to extract XML content
            pass
        
        # Parse XML structure (simplified)
        result = {
            'pages': [],
            'shapes': [],
            'connectors': []
        }
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract namespace if present
            ns = {'': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
            
            # Process shapes (placeholder)
            for shape in root.findall('.//Shape', ns):
                shape_data = {
                    'id': shape.get('ID', ''),
                    'name': shape.get('Name', ''),
                    'type': shape.get('Type', ''),
                    # Would extract position, size, text, etc.
                }
                result['shapes'].append(shape_data)
                
            # Process connectors (placeholder)
            for connector in root.findall('.//Connect', ns):
                connector_data = {
                    'from_sheet': connector.get('FromSheet', ''),
                    'to_sheet': connector.get('ToSheet', ''),
                    # Would extract connection details
                }
                result['connectors'].append(connector_data)
                
        except ET.ParseError as e:
            print(f"Error parsing Visio XML: {e}")
        
        return result


class RasterBlueprintReader(BlueprintReader):
    """Reader for raster image blueprints (JPG, PNG, etc.)"""
    
    def read(self, file_path):
        """Read raster blueprint image and extract content via OCR"""
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to load image from {file_path}")
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Extract text with OCR
        text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        
        # Extract lines and shapes (simplified)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        result = {
            'text_blocks': [],
            'lines': [],
            'image_size': img.shape
        }
        
        # Process OCR text results
        for i, text in enumerate(text_data['text']):
            if text.strip():
                result['text_blocks'].append({
                    'text': text.strip(),
                    'x': text_data['left'][i],
                    'y': text_data['top'][i],
                    'width': text_data['width'][i],
                    'height': text_data['height'][i],
                    'conf': text_data['conf'][i]
                })
        
        # Process detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result['lines'].append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
        
        return result


class SinergySimulationReader:
    """Reader for Sinergym simulation data"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def read(self, file_path):
        """Read Sinergym simulation results"""
        # Determine file type and read accordingly
        if file_path.endswith('.csv'):
            return self._read_csv(file_path)
        elif file_path.endswith('.json'):
            return self._read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")
    
    def _read_csv(self, file_path):
        """Read CSV simulation results"""
        df = pd.read_csv(file_path)
        # Process dataframe
        return {
            'time_series': df.to_dict(orient='records'),
            'columns': df.columns.tolist()
        }
    
    def _read_json(self, file_path):
        """Read JSON simulation results"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


class StructuralAnalysisReader:
    """Reader for structural analysis results"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def read(self, file_path):
        """Read structural analysis results from various formats"""
        # Read based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return self._read_csv(file_path)
        elif ext == '.json':
            return self._read_json(file_path)
        elif ext in ['.xml', '.msh']:
            return self._read_xml(file_path)
        else:
            raise ValueError(f"Unsupported structural analysis file format: {ext}")
    
    def _read_csv(self, file_path):
        """Read CSV structural results"""
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    
    def _read_json(self, file_path):
        """Read JSON structural results"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _read_xml(self, file_path):
        """Read XML structural results"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Process XML (simplified)
        result = {
            'nodes': [],
            'elements': [],
            'results': []
        }
        
        # Would extract nodes, elements, and analysis results
        
        return result


def get_reader_for_file(file_path, config=None):
    """Factory function to get appropriate reader based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return PDFBlueprintReader(config)
    elif ext in ['.dxf', '.dwg']:
        return CADBlueprintReader(config)
    elif ext in ['.vsd', '.vsdx', '.vdx']:
        return VisioXMLReader(config)
    elif ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
        return RasterBlueprintReader(config)
    elif ext in ['.csv', '.json'] and 'sinergy' in file_path.lower():
        return SinergySimulationReader(config)
    elif ext in ['.csv', '.json', '.xml', '.msh'] and 'struct' in file_path.lower():
        return StructuralAnalysisReader(config)
    else:
        raise ValueError(f"No appropriate reader found for file: {file_path}")