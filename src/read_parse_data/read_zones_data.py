"""
Read the JSON file that captures information about various zones in the facility and understand.
TODO: Add a flag to return data if this is to be used for training.
"""

import os
import json
import argparse
import traceback


def validate_epjson_file(file_path):
    """
    Validate the file used to define building models for energy simulation, geometry, construction and others.
    Args:
        file_path (.epJSON): Path to the file.
    """
    print(f"Validating epJSON file: {file_path}")

    # Check file existence
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    # Check file extension
    if not file_path.lower().endswith(".epjson"):
        print(f"Warning: File {file_path} does not have .epjson extension.")

    try:
        # Load the epJSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("\nSuccessfully loaded JSON data from file")

        # Check top-level structure
        print("\nChecking epJSON structure...")

        # Check for required high-level objects
        required_objects = [
            "Version",
            "Building",
            "SimulationControl",
            "Site:Location",
            "RunPeriod",
            "SizingPeriod:DesignDay",
            "GlobalGeometryRules",
        ]

        missing_objects = [obj for obj in required_objects if obj not in data]
        if missing_objects:
            print(f"Warning: Missing required objects: {missing_objects}")
        else:
            print("All required high-level objects are present")

        # Extract basic building information
        if "Building" in data:
            building_props = list(data["Building"].values())[0]
            print(f"\nBuilding name: {building_props.get('name', 'Not specified')}")
            print(f"North axis: {building_props.get('north_axis', 0)} degrees")
            print(f"Terrain: {building_props.get('terrain', 'Not specified')}")

        # Check location information
        if "Site:Location" in data:
            location_props = list(data["Site:Location"].values())[0]
            print(f"\nLocation: {location_props.get('name', 'Not specified')}")
            print(f"Latitude: {location_props.get('latitude', 'Not specified')}")
            print(f"Longitude: {location_props.get('longitude', 'Not specified')}")
            print(f"Time Zone: {location_props.get('time_zone', 'Not specified')}")
            print(f"Elevation: {location_props.get('elevation', 'Not specified')}")

        # Check simulation control settings
        if "SimulationControl" in data:
            sim_props = list(data["SimulationControl"].values())[0]
            print("\nSimulation Control:")
            print(f"Do zone sizing: {sim_props.get('do_zone_sizing_calculation', 'Not specified')}")
            print(f"Do system sizing: {sim_props.get('do_system_sizing_calculation', 'Not specified')}")
            print(f"Do plant sizing: {sim_props.get('do_plant_sizing_calculation', 'Not specified')}")
            print(
                f"Run simulation for sizing periods: {sim_props.get('run_simulation_for_sizing_periods', 'Not specified')}"
            )
            print(f"Run simulation for weather file: {sim_props.get('run_simulation_for_weather_file_run_periods', 'None')}")

        # Check run period
        if "RunPeriod" in data:
            run_props = list(data["RunPeriod"].values())[0]
            print("\nRun Period:")
            print(f"Begin month: {run_props.get('begin_month', 'Not specified')}")
            print(f"Begin day: {run_props.get('begin_day_of_month', 'Not specified')}")
            print(f"End month: {run_props.get('end_month', 'Not specified')}")
            print(f"End day: {run_props.get('end_day_of_month', 'Not specified')}")

        # Count objects by type
        print("\nObject counts by type:")
        for obj_type, objects in data.items():
            print(f"{obj_type}: {len(objects)}")

        # Check for zones
        if "Zone" in data:
            zones = data["Zone"]
            print(f"\nFound {len(zones)} thermal zones:")
            for zone_id, zone_props in zones.items():
                print(f"- {zone_props.get('name', zone_id)}")
        else:
            print("\nWarning: No thermal zones defined")

        # Check for HVAC systems
        hvac_objects = [
            "AirLoopHVAC",
            "ZoneHVAC:EquipmentConnections",
            "AirTerminal:SingleDuct:VAV:Reheat",
            "Boiler:HotWater",
            "Chiller:Electric",
            "Coil:Cooling:DX",
        ]

        hvac_count = sum(len(data.get(obj, {})) for obj in hvac_objects)
        if hvac_count > 0:
            print(f"\nFound approximately {hvac_count} HVAC-related objects")
        else:
            print("\nWarning: No HVAC systems defined")

        # Check for schedules
        schedule_types = [key for key in data.keys() if key.startswith("Schedule")]
        schedule_count = sum(len(data.get(schedule, {})) for schedule in schedule_types)
        if schedule_count > 0:
            print(f"\nFound {schedule_count} schedule objects across {len(schedule_types)} schedule types")
        else:
            print("\nWarning: No schedules defined")

        # Check for materials and constructions
        material_types = [
            "Material",
            "Material:NoMass",
            "Material:AirGap",
            "WindowMaterial:SimpleGlazingSystem",
        ]
        material_count = sum(len(data.get(material, {})) for material in material_types)

        construction_count = len(data.get("Construction", {}))

        if material_count > 0 and construction_count > 0:
            print(f"\nFound {material_count} material objects and {construction_count} constructions")
        else:
            print(f"\nWarning: Missing materials ({material_count}) or constructions ({construction_count})")

        # Check for internal loads
        load_objects = [
            "People",
            "Lights",
            "ElectricEquipment",
            "GasEquipment",
            "OtherEquipment",
            "ZoneInfiltration:DesignFlowRate",
        ]

        load_count = sum(len(data.get(obj, {})) for obj in load_objects)
        if load_count > 0:
            print(f"\nFound approximately {load_count} internal load objects")
        else:
            print("\nWarning: No internal loads defined")

        # Check for output variables
        if "Output:Variable" in data:
            output_vars = data["Output:Variable"]
            print(f"\nFound {len(output_vars)} output variable requests")
            if len(output_vars) < 5:
                print("Warning: Limited number of output variables defined")
        else:
            print("\nWarning: No output variables defined")

        # Check for meters
        if "Output:Meter" in data:
            meters = data["Output:Meter"]
            print(f"Found {len(meters)} output meter requests")

        # Create visualization of zone relationships (simplified)
        if "Zone" in data and "BuildingSurface:Detailed" in data:
            zones = data["Zone"]
            surfaces = data["BuildingSurface:Detailed"]

            # Create a dictionary to track zone connections
            zone_connections = {}
            for zone_id in zones.keys():
                zone_connections[zone_id] = []

            # Identify connections between zones through surfaces
            for surface_id, surface_props in surfaces.items():
                if (
                    "outside_boundary_condition" in surface_props
                    and surface_props["outside_boundary_condition"] == "Surface"
                ):
                    if "outside_boundary_condition_object" in surface_props:
                        this_zone = surface_props.get("zone_name", "")
                        other_surface = surface_props["outside_boundary_condition_object"]
                        # Find the zone of the other surface
                        for other_id, other_props in surfaces.items():
                            if other_props.get("name", "") == other_surface:
                                other_zone = other_props.get("zone_name", "")
                                if this_zone and other_zone and this_zone != other_zone:
                                    if other_zone not in zone_connections[this_zone]:
                                        zone_connections[this_zone].append(other_zone)

            # Create a simple visualization of zone connections
            # This is a very simplified approach - a proper visualization would require a graph library
            print("\nZone adjacency analysis:")
            for zone_id, connected_zones in zone_connections.items():
                zone_name = zones[zone_id].get("name", zone_id)
                if connected_zones:
                    connected_names = [zones[z_id].get("name", z_id) if z_id in zones else z_id for z_id in connected_zones]
                    print(f"{zone_name} is connected to: {', '.join(connected_names)}")
                else:
                    print(f"{zone_name} has no connections to other zones")

        print("\nValidation complete.")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}")
        print(f"JSON error: {str(e)}")
    except Exception as e:
        print(f"Error validating epJSON file: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate EnergyPlus JSON (epJSON) file")
    parser.add_argument("file_path", type=str, help="Path to the epJSON file")
    args = parser.parse_args()

    validate_epjson_file(args.file_path)
