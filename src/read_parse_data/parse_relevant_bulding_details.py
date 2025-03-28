"""
The epJSON file contains all kinds of details about the facility.
The key is to parse the relevant details and feed it to the LLM.
Because we don't want to overwhelm the system with too many irrelevant details
"""
import pprint
import argparse

import json
import pandas as pd
import numpy as np

def extract_zone_features(epjson_file, zone_name="SPACE1-1"):
    """
    Extract key features from an epJSON file for a specific zone.
    
    Parameters:
    -----------
    epjson_file : str
        Path to the epJSON file
    zone_name : str
        Name of the zone to extract information for
    
    Returns:
    --------
    dict
        Dictionary containing key features of the specified zone
    """
    # Load the epJSON file
    with open(epjson_file, 'r') as f:
        data = json.load(f)
    
    # Initialize the results dictionary
    results = {
        "zone_properties": {},
        "construction": {},
        "hvac_system": {},
        "control_parameters": {},
        "occupancy_loads": {}
    }
    
    # Extract Zone-Specific Properties
    if zone_name in data.get("Zone", {}):
        zone_info = data["Zone"][zone_name]
        results["zone_properties"] = {
            "ceiling_height": zone_info.get("ceiling_height", "N/A"),
            "volume": zone_info.get("volume", "N/A"),
            "direction_north": zone_info.get("direction_of_relative_north", "N/A"),
            "multiplier": zone_info.get("multiplier", "N/A"),
            "type": zone_info.get("type", "N/A")
        }
    
    # Find surfaces associated with this zone
    surfaces = []
    adjacencies = []
    exterior_surfaces = []
    
    for surf_name, surf_data in data.get("BuildingSurface:Detailed", {}).items():
        if surf_data.get("zone_name") == zone_name:
            surface_type = surf_data.get("surface_type", "Unknown")
            outside_boundary = surf_data.get("outside_boundary_condition", "Unknown")
            
            surfaces.append({
                "name": surf_name,
                "type": surface_type,
                "boundary": outside_boundary
            })
            
            # Check if it's an exterior surface
            if outside_boundary == "Outdoors":
                exterior_surfaces.append({
                    "name": surf_name,
                    "type": surface_type,
                    "sun_exposure": surf_data.get("sun_exposure", "Unknown"),
                    "wind_exposure": surf_data.get("wind_exposure", "Unknown")
                })
            
            # Check for adjacencies
            if outside_boundary == "Surface":
                adjacent_surface = surf_data.get("outside_boundary_condition_object", "Unknown")
                # Find the zone that contains this adjacent surface
                adjacent_zone = None
                for other_surf, other_data in data.get("BuildingSurface:Detailed", {}).items():
                    if other_surf == adjacent_surface:
                        adjacent_zone = other_data.get("zone_name", "Unknown")
                
                if adjacent_zone and adjacent_zone != zone_name:
                    adjacencies.append({
                        "adjacent_zone": adjacent_zone,
                        "surface_name": surf_name,
                        "adjacent_surface": adjacent_surface
                    })
    
    results["zone_properties"]["surfaces"] = surfaces
    results["zone_properties"]["adjacencies"] = adjacencies
    results["zone_properties"]["exterior_surfaces"] = exterior_surfaces
    
    # Extract windows/glazing for this zone
    windows = []
    for window_name, window_data in data.get("FenestrationSurface:Detailed", {}).items():
        parent_surface = window_data.get("building_surface_name", "")
        # Check if the parent surface belongs to our zone
        for surf in surfaces:
            if surf["name"] == parent_surface:
                construction_name = window_data.get("construction_name", "Unknown")
                windows.append({
                    "name": window_name,
                    "parent_surface": parent_surface,
                    "construction": construction_name,
                    "vf_to_ground": window_data.get("view_factor_to_ground", "N/A")
                })
    
    results["construction"]["windows"] = windows
    
    # Extract construction materials
    constructions = {}
    # Find all construction names used in the zone's surfaces
    used_constructions = set()
    for surf in surfaces:
        for surf_name, surf_data in data.get("BuildingSurface:Detailed", {}).items():
            if surf_name == surf["name"]:
                used_constructions.add(surf_data.get("construction_name", ""))
    
    for window in windows:
        used_constructions.add(window.get("construction", ""))
    
    # Get details of these constructions
    for const_name in used_constructions:
        if const_name and const_name in data.get("Construction", {}):
            const_data = data["Construction"][const_name]
            constructions[const_name] = const_data
    
    results["construction"]["materials"] = constructions
    
    # Extract infiltration data
    infiltration = {}
    for infil_name, infil_data in data.get("ZoneInfiltration:DesignFlowRate", {}).items():
        if zone_name in infil_data.get("zone_or_zonelist_or_space_or_spacelist_name", ""):
            infiltration = {
                "design_flow_rate": infil_data.get("design_flow_rate", "N/A"),
                "calculation_method": infil_data.get("design_flow_rate_calculation_method", "N/A"),
                "schedule": infil_data.get("schedule_name", "N/A"),
                "velocity_coefficient": infil_data.get("velocity_term_coefficient", "N/A")
            }
    
    results["construction"]["infiltration"] = infiltration
    
    # Extract HVAC System information
    hvac_equipment = {}
    # Find zone equipment connections
    for conn_name, conn_data in data.get("ZoneHVAC:EquipmentConnections", {}).items():
        if conn_data.get("zone_name") == zone_name:
            equipment_list_name = conn_data.get("zone_conditioning_equipment_list_name", "")
            if equipment_list_name and equipment_list_name in data.get("ZoneHVAC:EquipmentList", {}):
                equipments = data["ZoneHVAC:EquipmentList"][equipment_list_name].get("equipment", [])
                for equip in equipments:
                    equip_type = equip.get("zone_equipment_object_type", "")
                    equip_name = equip.get("zone_equipment_name", "")
                    
                    # Terminal unit details
                    if equip_type == "ZoneHVAC:AirDistributionUnit" and equip_name in data.get("ZoneHVAC:AirDistributionUnit", {}):
                        adu_data = data["ZoneHVAC:AirDistributionUnit"][equip_name]
                        terminal_type = adu_data.get("air_terminal_object_type", "")
                        terminal_name = adu_data.get("air_terminal_name", "")
                        
                        if terminal_type and terminal_name in data.get(terminal_type, {}):
                            terminal_data = data[terminal_type][terminal_name]
                            hvac_equipment["terminal_unit"] = {
                                "type": terminal_type,
                                "name": terminal_name,
                                "max_flow_rate": terminal_data.get("maximum_air_flow_rate", "N/A"),
                                "min_flow_fraction": terminal_data.get("constant_minimum_air_flow_fraction", "N/A")
                            }
                            
                            # Reheat coil info if available
                            reheat_coil_type = terminal_data.get("reheat_coil_object_type", "")
                            reheat_coil_name = terminal_data.get("reheat_coil_name", "")
                            
                            if reheat_coil_type and reheat_coil_name in data.get(reheat_coil_type, {}):
                                coil_data = data[reheat_coil_type][reheat_coil_name]
                                hvac_equipment["reheat_coil"] = {
                                    "type": reheat_coil_type,
                                    "name": reheat_coil_name,
                                    "efficiency": coil_data.get("efficiency", "N/A"),
                                    "capacity": coil_data.get("nominal_capacity", "N/A")
                                }
    
    # Get air loop info
    air_loop_name = None
    for splitter_name, splitter_data in data.get("AirLoopHVAC:ZoneSplitter", {}).items():
        nodes = splitter_data.get("nodes", [])
        for node in nodes:
            node_name = node.get("outlet_node_name", "")
            if node_name.startswith(f"{zone_name} ATU"):
                air_loop_name = "VAV Sys 1"  # This is known from examining the file structure
    
    if air_loop_name and air_loop_name in data.get("AirLoopHVAC", {}):
        air_loop_data = data["AirLoopHVAC"][air_loop_name]
        hvac_equipment["air_loop"] = {
            "name": air_loop_name,
            "design_flow_rate": air_loop_data.get("design_supply_air_flow_rate", "N/A")
        }
        
    results["hvac_system"] = hvac_equipment
    
    # Extract Control Parameters
    # Thermostat control
    thermostat_info = {}
    for ctrl_name, ctrl_data in data.get("ZoneControl:Thermostat", {}).items():
        if ctrl_data.get("zone_or_zonelist_name") == zone_name:
            thermostat_info["control_name"] = ctrl_name
            thermostat_info["control_type_schedule"] = ctrl_data.get("control_type_schedule_name", "N/A")
            
            # Get setpoint objects
            control_objects = []
            for i in range(1, 4):  # Check for up to 3 control objects
                obj_type = ctrl_data.get(f"control_{i}_object_type", None)
                obj_name = ctrl_data.get(f"control_{i}_name", None)
                
                if obj_type and obj_name:
                    if obj_name in data.get(obj_type, {}):
                        setpoint_data = data[obj_type][obj_name]
                        control_objects.append({
                            "type": obj_type,
                            "name": obj_name,
                            "schedule": setpoint_data.get("setpoint_temperature_schedule_name", 
                                        setpoint_data.get("cooling_setpoint_temperature_schedule_name", 
                                        setpoint_data.get("heating_setpoint_temperature_schedule_name", "N/A")))
                        })
            
            thermostat_info["control_objects"] = control_objects
    
    # Extract schedule details for setpoints
    setpoint_schedules = {}
    for control_obj in thermostat_info.get("control_objects", []):
        schedule_name = control_obj.get("schedule", "")
        if schedule_name and schedule_name in data.get("Schedule:Compact", {}):
            schedule_data = data["Schedule:Compact"][schedule_name]
            # Extract the key setpoint values (simplified)
            setpoint_values = []
            for i in range(0, len(schedule_data.get("data", [])), 4):  # Approximate parsing of schedule data
                if i+3 < len(schedule_data.get("data", [])):
                    time_spec = schedule_data["data"][i+2].get("field", "")
                    value = schedule_data["data"][i+3].get("field", "")
                    
                    # Ensure time_spec is a string before calling startswith()
                    if isinstance(time_spec, str) and time_spec.startswith("Until:") and value:
                        try:
                            value_float = float(value)
                            setpoint_values.append({
                                "time": time_spec.replace("Until:", "").strip(),
                                "value": value_float
                            })
                        except (ValueError, TypeError):
                            pass
        
        setpoint_schedules[schedule_name] = setpoint_values
    
    results["control_parameters"]["thermostat"] = thermostat_info
    results["control_parameters"]["setpoint_schedules"] = setpoint_schedules
    
    # Extract Occupancy and Internal Loads
    # People
    people_info = {}
    for people_name, people_data in data.get("People", {}).items():
        if people_data.get("zone_or_zonelist_or_space_or_spacelist_name") == zone_name:
            people_info = {
                "name": people_name,
                "number_of_people": people_data.get("number_of_people", "N/A"),
                "schedule": people_data.get("number_of_people_schedule_name", "N/A"),
                "activity_level_schedule": people_data.get("activity_level_schedule_name", "N/A"),
                "fraction_radiant": people_data.get("fraction_radiant", "N/A")
            }
    
    # Lights
    lights_info = {}
    for light_name, light_data in data.get("Lights", {}).items():
        if light_data.get("zone_or_zonelist_or_space_or_spacelist_name") == zone_name:
            lights_info = {
                "name": light_name,
                "design_level": light_data.get("lighting_level", "N/A"),
                "schedule": light_data.get("schedule_name", "N/A"),
                "fraction_radiant": light_data.get("fraction_radiant", "N/A"),
                "return_air_fraction": light_data.get("return_air_fraction", "N/A")
            }
    
    # Equipment
    equipment_info = {}
    for eq_name, eq_data in data.get("ElectricEquipment", {}).items():
        if eq_data.get("zone_or_zonelist_or_space_or_spacelist_name") == zone_name:
            equipment_info = {
                "name": eq_name,
                "design_level": eq_data.get("design_level", "N/A"),
                "schedule": eq_data.get("schedule_name", "N/A"),
                "fraction_radiant": eq_data.get("fraction_radiant", "N/A"),
                "fraction_latent": eq_data.get("fraction_latent", "N/A")
            }
    
    results["occupancy_loads"]["people"] = people_info
    results["occupancy_loads"]["lights"] = lights_info
    results["occupancy_loads"]["equipment"] = equipment_info
    
    return results

# Usage example:
# features = extract_zone_features('sample_epJSON.json', 'SPACE1-1')


if __name__ == "__main__":
    # Setup the command line for argument parsing
    parser = argparse.ArgumentParser(description="Extract features from an epJSON")
    parser.add_argument("epJSON_file", type=str, help="Path to the epJSON file")
    parser.add_argument("--zone", type=str, default="SPACE1-1", help="Name of the zone")

    # Parse the arguments
    args = parser.parse_args()

    # Extract the features
    features = extract_zone_features(args.epJSON_file, args.zone)

    # Print the results
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(features)
