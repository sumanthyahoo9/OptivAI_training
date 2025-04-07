"""
This script helps us read a blueprint in the pdf format and get it
ready for OCR models.
The facility layout is critical since it adds a lot of context beyond
raw numbers for the LLM to make sense of the agent's actions.
This script assumes that the blueprints are in the PDF format.
Once we extract the blueprint using OCR, we can integrate it
with the epJSON file with the data about the zones
"""
import os
import argparse
import json
import re
from tqdm import tqdm
from PIL import Image
import numpy as np
import easyocr

def load_blueprints(blueprint_dir):
    """
    Load blueprint images from directory
    """
    print(f"Loading blueprint images from {blueprint_dir}...")
    images = []
    image_paths = []
    
    if not os.path.exists(blueprint_dir):
        raise FileNotFoundError(f"Blueprint directory {blueprint_dir} not found")
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.pdf']
    
    for file in os.listdir(blueprint_dir):
        file_path = os.path.join(blueprint_dir, file)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            try:
                img = Image.open(file_path)
                images.append(img)
                image_paths.append(file_path)
                print(f"  Loaded {file_path}, size: {img.size}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    print(f"Loaded {len(images)} blueprint images")
    return images, image_paths

def perform_ocr_easyocr(images, image_paths):
    """
    Perform OCR on blueprint images using EasyOCR
    """
    print("Initializing EasyOCR model...")
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English
    
    all_text = []
    
    for i, (img, img_path) in enumerate(tqdm(zip(images, image_paths), total=len(images), desc="Processing blueprints")):
        print(f"\nProcessing blueprint {i+1}/{len(images)}: {os.path.basename(img_path)}")
        
        # Convert PIL image to numpy array if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
            
        # Perform OCR
        try:
            results = reader.readtext(img_np)
            
            # Extract text from results
            page_text = []
            for detection in results:
                # Extract text and confidence
                bbox, text, confidence = detection
                page_text.append(text)
                
            # Join all detected text for this blueprint
            blueprint_text = "\n".join(page_text)
            
            # Add blueprint name as header
            blueprint_name = os.path.basename(img_path)
            all_text.append(f"# BLUEPRINT: {blueprint_name}\n\n{blueprint_text}\n\n")
            
            print(f"  Extracted {len(page_text)} text elements")
            
        except Exception as e:
            print(f"  Error performing OCR on {img_path}: {e}")
    
    # Combine all extracted text
    combined_text = "\n".join(all_text)
    
    return combined_text

def post_process_ocr(ocr_text):
    """
    Post-process OCR text to improve quality
    """
    # Replace common OCR errors
    replacements = {
        # Common OCR errors in technical drawings
        'O': '0',  # Replace letter O with number 0 in specific contexts
        'l': '1',  # Replace letter l with number 1 in specific contexts
    }
    
    processed_text = ocr_text
    
    # Apply specific replacements in numeric contexts
    for old, new in replacements.items():
        # Only replace in numeric contexts, e.g., "l0.5" -> "10.5"
        processed_text = re.sub(r'(\d)' + old + r'(\d)', r'\1' + new + r'\2', processed_text)
        processed_text = re.sub(r'^' + old + r'(\d)', new + r'\1', processed_text, flags=re.MULTILINE)
        processed_text = re.sub(r'(\d)' + old + r'$', r'\1' + new, processed_text, flags=re.MULTILINE)
    
    # Normalize whitespace
    processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
    
    # Extract zone information
    zone_matches = re.finditer(r'(ZONE|SPACE)[- ]?([A-Za-z0-9-]+)', processed_text, re.IGNORECASE)
    zones = {}
    
    for match in zone_matches:
        zone_type = match.group(1)
        zone_id = match.group(2)
        zones[zone_id] = {"type": zone_type}
    
    # Extract HVAC component information
    hvac_patterns = [
        (r'(HVAC|AirLoop)[- ]?([A-Za-z0-9-]+)', "system"),
        (r'(Cooling|Heating)[- ]?Coil[- ]?([A-Za-z0-9-]+)', "coil"),
        (r'Fan[- ]?([A-Za-z0-9-]+)', "fan"),
        (r'VAV[- ]?([A-Za-z0-9-]+)', "vav")
    ]
    
    hvac_components = []
    
    for pattern, component_type in hvac_patterns:
        matches = re.finditer(pattern, processed_text, re.IGNORECASE)
        for match in matches:
            component_id = match.group(2) if len(match.groups()) > 1 else match.group(1)
            hvac_components.append({
                "type": component_type,
                "id": component_id
            })
    
    # Structure extraction results
    extraction_results = {
        "zones": zones,
        "hvac_components": hvac_components
    }
    
    return processed_text, extraction_results

def format_extraction_results(extraction_results):
    """
    Format extraction results into a readable text summary
    """
    formatted_text = []
    
    # Format zones information
    formatted_text.append("BUILDING ZONES IDENTIFIED:")
    formatted_text.append("==========================")
    
    if extraction_results["zones"]:
        for zone_id, zone_info in extraction_results["zones"].items():
            formatted_text.append(f"- {zone_info['type']} {zone_id}")
    else:
        formatted_text.append("No zones identified.")
    
    # Format HVAC components information
    formatted_text.append("\nHVAC COMPONENTS IDENTIFIED:")
    formatted_text.append("===========================")
    
    if extraction_results["hvac_components"]:
        component_types = {}
        for component in extraction_results["hvac_components"]:
            component_type = component["type"]
            if component_type not in component_types:
                component_types[component_type] = []
            component_types[component_type].append(component["id"])
        
        for component_type, component_ids in component_types.items():
            formatted_text.append(f"\n{component_type.upper()} COMPONENTS:")
            for component_id in component_ids:
                formatted_text.append(f"- {component_id}")
    else:
        formatted_text.append("No HVAC components identified.")
    
    return "\n".join(formatted_text)

def main():
    """
    Run this script to extract the blueprint from the pdf or image files
    """
    parser = argparse.ArgumentParser(description="Extract text from blueprint images using OCR")
    parser.add_argument("--blueprint_dir", type=str, required=True, help="Directory containing blueprint images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--raw_output", type=str, default="blueprint_ocr_raw.txt", help="Raw OCR output file name")
    parser.add_argument("--processed_output", type=str, default="blueprint_ocr.txt", help="Processed OCR output file name")
    parser.add_argument("--json_output", type=str, default="blueprint_extraction.json", help="JSON extraction results file name")
    parser.add_argument("--summary_output", type=str, default="blueprint_summary.txt", help="Summary text file name")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load blueprint images
    images, image_paths = load_blueprints(args.blueprint_dir)
    
    if not images:
        print("No blueprint images found. Exiting.")
        return
    
    # Perform OCR
    ocr_text = perform_ocr_easyocr(images, image_paths)
    
    # Save raw OCR text
    raw_output_path = os.path.join(args.output_dir, args.raw_output)
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    print(f"Raw OCR text saved to {raw_output_path}")
    
    # Post-process OCR text
    processed_text, extraction_results = post_process_ocr(ocr_text)
    
    # Save processed OCR text
    processed_output_path = os.path.join(args.output_dir, args.processed_output)
    with open(processed_output_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    print(f"Processed OCR text saved to {processed_output_path}")
    
    # Save extraction results as JSON
    json_output_path = os.path.join(args.output_dir, args.json_output)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(extraction_results, f, indent=2)
    print(f"Extraction results saved to {json_output_path}")
    
    # Format and save summary
    summary_text = format_extraction_results(extraction_results)
    summary_output_path = os.path.join(args.output_dir, args.summary_output)
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Summary saved to {summary_output_path}")

if __name__ == "__main__":
    main()
