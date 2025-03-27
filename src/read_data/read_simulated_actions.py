"""
This script reads the simulated actions of the agent
This goes hand-in-hand with the script that reads the agent actions
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def validate_simulated_actions(file_path):
    """
    Validate the simulated actions of the agent
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
    
    # Check for constant values
    print("\nChecking for constant values...")
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values == 1:
            print(f"Warning: Column {col} has only one value: {df[col].iloc[0]}")
        else:
            print(f"Column {col} has {unique_values} unique values")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    plt.figure(figsize=(12, 8))
    
    # Time series plot of setpoints
    plt.subplot(2, 1, 1)
    plt.plot(df['Heating_Setpoint_RL'], label='Heating Setpoint')
    plt.plot(df['Cooling_Setpoint_RL'], label='Cooling Setpoint')
    plt.title('Setpoint Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    # Scatter plot to visualize relationship between heating and cooling setpoints
    plt.subplot(2, 1, 2)
    plt.scatter(df['Heating_Setpoint_RL'], df['Cooling_Setpoint_RL'], alpha=0.5)
    plt.title('Heating vs Cooling Setpoints')
    plt.xlabel('Heating Setpoint (°C)')
    plt.ylabel('Cooling Setpoint (°C)')
    
    # Add a reference line for equal values
    min_val = min(df['Heating_Setpoint_RL'].min(), df['Cooling_Setpoint_RL'].min())
    max_val = max(df['Heating_Setpoint_RL'].max(), df['Cooling_Setpoint_RL'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Equal Values Line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulated_actions_analysis.png')
    print("Visualizations saved to 'simulated_actions_analysis.png'")
    
    # Compare with agent_actions.csv if it exists
    try:
        agent_df = pd.read_csv('agent_actions.csv')
        if agent_df.shape == df.shape:
            print("\nComparing with agent_actions.csv...")
            
            # Check if they're identical
            if agent_df.equals(df):
                print("simulated_actions.csv is identical to agent_actions.csv")
            else:
                # Calculate differences
                diff_heating = (agent_df['Heating_Setpoint_RL'] - df['Heating_Setpoint_RL']).abs()
                diff_cooling = (agent_df['Cooling_Setpoint_RL'] - df['Cooling_Setpoint_RL']).abs()
                
                print("Heating setpoint differences (absolute):")
                print(f"  Mean: {diff_heating.mean():.4f}°C")
                print(f"  Max: {diff_heating.max():.4f}°C")
                print(f"  Number of differences: {(diff_heating > 0).sum()}")
                
                print("Cooling setpoint differences (absolute):")
                print(f"  Mean: {diff_cooling.mean():.4f}°C")
                print(f"  Max: {diff_cooling.max():.4f}°C")
                print(f"  Number of differences: {(diff_cooling > 0).sum()}")
    except Exception as e:
        print(f"\nUnable to compare with agent_actions.csv: {e}")
    
    print("\nValidation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate simulated actions CSV file')
    parser.add_argument('file_path', type=str, help='Path to the simulated actions CSV file')
    args = parser.parse_args()
    validate_simulated_actions(args.file_path)