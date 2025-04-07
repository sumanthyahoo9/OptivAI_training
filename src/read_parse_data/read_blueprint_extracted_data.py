"""
This script helps make sense of the extracted OCR data from a blueprint
"""
import re
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def clean_ocr_text(text):
    """
    Clean OCR-extracted text by:
    1. Removing excessive whitespace
    2. Fixing common OCR errors
    3. Restructuring into a more usable format
    """
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors specific to building blueprints
    text = text.replace('0cC', '°C')  # Fix for degree Celsius
    text = text.replace('rn2', 'm²')   # Fix for square meters
    text = text.replace('rn3', 'm³')   # Fix for cubic meters
    
    # Additional cleaning specific to blueprint text
    # (Add more as needed based on your OCR output patterns)
    
    return text.strip()

def extract_zones_info(text):
    """
    Extract information about building zones from OCR text
    Returns a structured dictionary of zone information
    """
    zones = {}
    
    # Look for zone patterns
    zone_pattern = r'(?:SPACE|ZONE)[- ]?(\w+[-\d]*)'
    zone_matches = re.finditer(zone_pattern, text, re.IGNORECASE)
    
    for match in zone_matches:
        zone_id = match.group(1)
        
        # Extract zone data (customize these patterns based on your blueprint content)
        area_match = re.search(rf'{zone_id}.*?area[: ]+([0-9.]+)\s*(?:m2|square meters)', text, re.IGNORECASE)
        volume_match = re.search(rf'{zone_id}.*?volume[: ]+([0-9.]+)\s*(?:m3|cubic meters)', text, re.IGNORECASE)
        height_match = re.search(rf'{zone_id}.*?height[: ]+([0-9.]+)\s*(?:m|meters)', text, re.IGNORECASE)
        
        zones[zone_id] = {
            "area": float(area_match.group(1)) if area_match else None,
            "volume": float(volume_match.group(1)) if volume_match else None,
            "height": float(height_match.group(1)) if height_match else None,
        }
    
    return zones

def extract_hvac_info(text):
    """
    Extract HVAC system information from OCR text
    Returns a structured dictionary of HVAC components
    """
    hvac_info = {
        "systems": [],
        "components": []
    }
    
    # Extract HVAC system names
    hvac_pattern = r'(?:HVAC|AirLoop)[- ]?(\w+[-\d]*)'
    hvac_matches = re.finditer(hvac_pattern, text, re.IGNORECASE)
    
    for match in hvac_matches:
        system_id = match.group(1)
        hvac_info["systems"].append(system_id)
    
    # Extract components like cooling coils, heating coils, etc.
    component_patterns = [
        (r'Cooling[- ]?Coil[- ]?(\w+[-\d]*)', "cooling_coil"),
        (r'Heating[- ]?Coil[- ]?(\w+[-\d]*)', "heating_coil"),
        (r'Fan[- ]?(\w+[-\d]*)', "fan"),
        (r'VAV[- ]?(\w+[-\d]*)', "vav")
    ]
    
    for pattern, component_type in component_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            component_id = match.group(1)
            hvac_info["components"].append({
                "id": component_id,
                "type": component_type
            })
    
    return hvac_info

def extract_material_info(text):
    """
    Extract material information from OCR text
    Returns a list of materials mentioned in the blueprint
    """
    materials = []
    
    # Look for material sections
    material_section = re.search(r'Material(?:.*?):(.*?)(?:Construction|$)', text, re.DOTALL | re.IGNORECASE)
    
    if material_section:
        material_text = material_section.group(1)
        
        # Extract individual materials
        material_pattern = r'"([^"]+)"'
        material_matches = re.finditer(material_pattern, material_text)
        
        for match in material_matches:
            materials.append(match.group(1))
    
    return materials

def process_blueprint_ocr(ocr_text_file, output_json_file=None, output_txt_file=None):
    """
    Process OCR-extracted text from building blueprints
    and convert it to structured data and a clean text format
    """
    try:
        # Read OCR text
        with open(ocr_text_file, 'r', encoding="utf-8") as f:
            ocr_text = f.read()
        
        # Clean the OCR text
        cleaned_text = clean_ocr_text(ocr_text)
        
        # Extract structured information
        zones_info = extract_zones_info(cleaned_text)
        hvac_info = extract_hvac_info(cleaned_text)
        materials = extract_material_info(cleaned_text)
        
        # Create structured data dictionary
        structured_data = {
            "zones": zones_info,
            "hvac": hvac_info,
            "materials": materials
        }
        
        # Create a cleaned and formatted text version for LLM context
        formatted_text = []
        formatted_text.append("BUILDING BLUEPRINT INFORMATION")
        formatted_text.append("==============================")
        
        # Add zones information
        formatted_text.append("\nBUILDING ZONES:")
        for zone_id, zone_data in zones_info.items():
            formatted_text.append(f"Zone {zone_id}:")
            if zone_data["area"]:
                formatted_text.append(f"  - Area: {zone_data['area']} m²")
            if zone_data["volume"]:
                formatted_text.append(f"  - Volume: {zone_data['volume']} m³")
            if zone_data["height"]:
                formatted_text.append(f"  - Height: {zone_data['height']} m")
        
        # Add HVAC information
        formatted_text.append("\nHVAC SYSTEMS:")
        for system in hvac_info["systems"]:
            formatted_text.append(f"- HVAC System: {system}")
        
        formatted_text.append("\nHVAC COMPONENTS:")
        for component in hvac_info["components"]:
            formatted_text.append(f"- {component['type'].replace('_', ' ').title()}: {component['id']}")
        
        # Add materials information
        if materials:
            formatted_text.append("\nBUILDING MATERIALS:")
            for material in materials:
                formatted_text.append(f"- {material}")
        
        # Add raw OCR text (optional, but useful for context)
        formatted_text.append("\nRAW BLUEPRINT INFORMATION:")
        formatted_text.append(cleaned_text[:2000])  # Limit to first 2000 chars to avoid too much noise
        
        # Save the structured data if output path is provided
        if output_json_file:
            with open(output_json_file, 'w', encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2)
            print(f"Saved structured blueprint data to {output_json_file}")
        
        # Save the formatted text if output path is provided
        if output_txt_file:
            with open(output_txt_file, 'w', encoding="utf-8") as f:
                f.write('\n'.join(formatted_text))
            print(f"Saved formatted blueprint text to {output_txt_file}")
        
        # Return both structured data and formatted text
        return structured_data, '\n'.join(formatted_text)
    
    except Exception as e:
        print(f"Error processing blueprint OCR: {e}")
        return None, None

def create_blueprint_context_from_epjson(epjson_file, output_txt_file):
    """
    Creates a text-based building context from an EnergyPlus JSON file (epJSON)
    """
    try:
        # For the sample_epJSON.pdf, we'll parse it as if it were a JSON file
        # In a real scenario, you would read the actual JSON file
        
        # Load JSON data
        with open(epjson_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        # Create context sections
        context_lines = []
        context_lines.append("BUILDING INFORMATION FROM ENERGYPLUS MODEL")
        context_lines.append("=========================================")
        
        # Extract building info
        if "Building" in data:
            buildings = data["Building"]
            context_lines.append("\nBUILDING DETAILS:")
            for bldg_id, bldg_data in buildings.items():
                context_lines.append(f"Building: {bldg_id}")
                
                # Add building properties
                for prop, value in bldg_data.items():
                    if prop != "name":  # Skip name as we already used it above
                        formatted_prop = prop.replace("_", " ").title()
                        context_lines.append(f"- {formatted_prop}: {value}")
        
        # Extract zone info
        if "Zone" in data:
            zones = data["Zone"]
            context_lines.append("\nZONES:")
            for zone_id, zone_data in zones.items():
                context_lines.append(f"Zone: {zone_id}")
                for prop, value in zone_data.items():
                    if prop != "name":  # Skip name as we already used it
                        formatted_prop = prop.replace("_", " ").title()
                        context_lines.append(f"- {formatted_prop}: {value}")
        
        # Extract HVAC system info
        hvac_types = ["AirLoopHVAC", "Fan:VariableVolume", "Coil:Heating:Electric", "Coil:Cooling:DX"]
        context_lines.append("\nHVAC SYSTEMS:")
        
        for hvac_type in hvac_types:
            if hvac_type in data:
                components = data[hvac_type]
                context_lines.append(f"\n{hvac_type} Components:")
                for comp_id, comp_data in components.items():
                    context_lines.append(f"- {comp_id}")
                    # Add a few key properties for each component
                    properties_to_show = 5  # Limit to prevent too verbose output
                    prop_count = 0
                    
                    for prop, value in comp_data.items():
                        if prop != "name" and prop_count < properties_to_show:
                            if isinstance(value, dict) or isinstance(value, list):
                                # Skip complex nested structures
                                continue
                            formatted_prop = prop.replace("_", " ").title()
                            context_lines.append(f"  * {formatted_prop}: {value}")
                            prop_count += 1
        
        # Extract construction and materials info
        if "Construction" in data:
            constructions = data["Construction"]
            context_lines.append("\nCONSTRUCTIONS:")
            for const_id, const_data in constructions.items():
                context_lines.append(f"- {const_id}")
                for prop, value in const_data.items():
                    if prop != "name":
                        formatted_prop = prop.replace("_", " ").title()
                        context_lines.append(f"  * {formatted_prop}: {value}")
        
        # Write the context to a file
        with open(output_txt_file, 'w', encoding="utf-8") as f:
            f.write('\n'.join(context_lines))
        
        print(f"Created blueprint context file: {output_txt_file}")
        return '\n'.join(context_lines)
    
    except Exception as e:
        print(f"Error creating blueprint context: {e}")
        return ""

def combine_blueprint_contexts(file_paths, output_file):
    """
    Combines multiple blueprint context files into a single comprehensive context
    """
    combined_context = []
    combined_context.append("COMBINED BUILDING BLUEPRINT INFORMATION")
    combined_context.append("=======================================")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                content = f.read()
                
            # Add a header for this file's content
            file_name = os.path.basename(file_path)
            combined_context.append(f"\n\n## INFORMATION FROM {file_name.upper()} ##")
            combined_context.append(content)
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Write the combined context to the output file
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write('\n'.join(combined_context))
    
    print(f"Created combined blueprint context file: {output_file}")
    return '\n'.join(combined_context)

def main():
    """
    Run this script to make sense of the data extracted using the OCR model
    """
    # Define file paths
    ocr_text_file = "blueprint_ocr.txt"  # OCR-extracted text from blueprints
    epjson_file = "sample_epJSON.pdf"    # Sample epJSON file (in real scenarios, use actual JSON)
    
    # Output paths
    structured_data_file = "blueprint_structured.json"
    ocr_context_file = "blueprint_context_ocr.txt"
    epjson_context_file = "blueprint_context_epjson.txt"
    combined_context_file = "blueprint_context.txt"  # Final context file for LLM fine-tuning
    
    # Process OCR text
    print("Processing OCR text from blueprints...")
    if os.path.exists(ocr_text_file):
        _, _ = process_blueprint_ocr(
            ocr_text_file=ocr_text_file,
            output_json_file=structured_data_file,
            output_txt_file=ocr_context_file
        )
    else:
        print(f"OCR text file {ocr_text_file} not found. Creating a placeholder...")
        # Create a placeholder OCR context
        with open(ocr_text_file, 'w', encoding="utf-8") as f:
            f.write("This is a placeholder for OCR-extracted blueprint text.\n")
            f.write("In a real scenario, this would contain text extracted from building blueprints.")
        
        process_blueprint_ocr(
            ocr_text_file=ocr_text_file,
            output_json_file=structured_data_file,
            output_txt_file=ocr_context_file
        )
    
    # Create context from epJSON
    print("Creating context from epJSON file...")
    if os.path.exists(epjson_file):
        try:
            create_blueprint_context_from_epjson(
                epjson_file=epjson_file, 
                output_txt_file=epjson_context_file
            )
        except Exception as e:
            print(f"Error processing epJSON file: {e}")
            # Create a placeholder epJSON context
            with open(epjson_context_file, 'w', encoding="utf-8") as f:
                f.write("This is a placeholder for epJSON-derived building context.\n")
                f.write("In a real scenario, this would contain structured information from the building model.")
    else:
        print(f"epJSON file {epjson_file} not found. Creating a placeholder...")
        # Create a placeholder epJSON context
        with open(epjson_context_file, 'w', encoding="utf-8") as f:
            f.write("This is a placeholder for epJSON-derived building context.\n")
            f.write("In a real scenario, this would contain structured information from the building model.")
    
    # Combine contexts
    print("Combining blueprint contexts...")
    combine_blueprint_contexts(
        file_paths=[ocr_context_file, epjson_context_file],
        output_file=combined_context_file
    )
    
    print("Blueprint context extraction complete!")

if __name__ == "__main__":
    main()
