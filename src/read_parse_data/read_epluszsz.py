"""
This script helps to understand make sense of the way the Energy Plus environment
works with our simulations
"""

import json
import argparse
import pandas as pd


def parse_epluszsz_data(file_path):
    """
    Read and parse the epluszsz data to extract SPACES-1 sizing data
    Args:
        file_path (_type_): _description_
    """
    space5_sizing_data = {}
    try:
        epluszsz_data = pd.read_csv(file_path)
        print(f"Successfully loaded the Energy Plus ZSZ csv with the shape: {epluszsz_data.shape}")
        # Filter column names for SPACE5-1
        space5_cols = [col for col in epluszsz_data.columns if "SPACE5-5" in col]
        if space5_cols:
            print(f"Found {len(space5_cols)} columns for SPACE5-5 in epluszsz.csv")
            # Get the last row of the data (typically contains the final sizing values)
            last_row = epluszsz_data.iloc[-1]
            # Extract key sizing parameters
            for col in space5_cols:
                # Create a simplified key name
                key = col.split(":")[-1].strip() if ":" in col else col
                key = key.replace("[", "_").replace("]", "").replace(" ", "_").lower()
                space5_sizing_data[key] = last_row[col]
            print(f"Extracted sizing data for SPACE5-5: {len(space5_sizing_data)} parameters")
        else:
            print("No data for the above zone was found")
    except Exception as e:
        print(f"Error loading or parsing epluszsz.csv: {e}")
    return space5_sizing_data


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Parse EnergyPlus Zone Sizing CSV file for SPACE5-5 data")
    parser.add_argument("file_path", type=str, help="Path to the epluszsz.csv file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path (optional)")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print the JSON output")
    args = parser.parse_args()

    # Parse the CSV file
    sizing_data = parse_epluszsz_data(args.file_path)
    # Output the results
    if len(sizing_data) > 0:
        if args.output:
            # Save to JSON file
            with open(args.output, "w", encoding="utf-8") as f:
                if args.pretty:
                    json.dump(sizing_data, f, indent=4)
                else:
                    json.dump(sizing_data, f)
            print(f"Saved SPACE5-5 sizing data to {args.output}")
        else:
            # Print to console
            if args.pretty:
                print(json.dumps(sizing_data, indent=4))
            else:
                print(sizing_data)

        print(f"Successfully extracted {len(sizing_data)} parameters for SPACE5-5")
    else:
        print("No SPACE5-5 data was found in the provided CSV file")
