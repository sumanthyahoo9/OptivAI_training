"""
This script reads the agent's actions throughout its operations
"""
import sys
import argparse
import pandas as pd


def validate_agent_actions(file_path):
    """
    Validate the agent's actions
    Args:
        file_path (_type_): _description_
    """
    # Load the CSV file
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check file dimensions
    print(f"\nFile dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for expected columns
    expected_columns = ["Heating_Setpoint_RL", "Cooling_Setpoint_RL"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
    else:
        print("All expected columns are present")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values detected")
    
    # Describe the data (including min, max, etc.)
    print("\nData summary:")
    print(df.describe())
    
    # Check for potential outliers based on expected ranges
    # Assuming heating setpoint should be between 15-25°C and cooling between 20-30°C
    heating_min, heating_max = 15, 25
    cooling_min, cooling_max = 20, 30
    
    # Check for heating setpoint outliers
    heating_outliers = df[(df["Heating_Setpoint_RL"] < heating_min) | 
                          (df["Heating_Setpoint_RL"] > heating_max)]
    if not heating_outliers.empty:
        print(f"\nFound {len(heating_outliers)} potential heating setpoint outliers")
        print(f"Sample outliers (first 5):")
        print(heating_outliers.head())
    else:
        print(f"\nNo heating setpoint outliers detected outside range [{heating_min}, {heating_max}]")
    
    # Check for cooling setpoint outliers
    cooling_outliers = df[(df["Cooling_Setpoint_RL"] < cooling_min) | 
                          (df["Cooling_Setpoint_RL"] > cooling_max)]
    if not cooling_outliers.empty:
        print(f"\nFound {len(cooling_outliers)} potential cooling setpoint outliers")
        print(f"Sample outliers (first 5):")
        print(cooling_outliers.head())
    else:
        print(f"\nNo cooling setpoint outliers detected outside range [{cooling_min}, {cooling_max}]")
    
    # Validate constraint: Heating setpoint should be less than cooling setpoint
    invalid_setpoints = df[df["Heating_Setpoint_RL"] >= df["Cooling_Setpoint_RL"]]
    if not invalid_setpoints.empty:
        print(f"\nFound {len(invalid_setpoints)} rows where heating setpoint >= cooling setpoint")
        print(f"Sample invalid setpoints (first 5):")
        print(invalid_setpoints.head())
    else:
        print("\nAll setpoints satisfy the constraint: heating < cooling")

# Run the validation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate CSV file")
    parser.add_argument('file_path', type=str, help="Path to the csv file for validation")
    args = parser.parse_args()

    if args.file_path.endswith("agent_actions.csv"):
        validate_agent_actions(args.file_path)
    else:
        print("Unsupported file format")
        sys.exit(1)