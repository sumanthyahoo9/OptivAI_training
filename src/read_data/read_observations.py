"""
This script helps us make sense of the various observations in the facility
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def validate_observations(file_path):
    """
    Validate the observations from the sensors and other devices prior to training.
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
    expected_columns = [
        "month",
        "day_of_month",
        "hour",
        "outdoor_temperature",
        "outdoor_humidity",
        "wind_speed",
        "wind_direction",
        "diffuse_solar_radiation",
        "direct_solar_radiation",
        "htg_setpoint",
        "clg_setpoint",
        "air_temperature",
        "air_humidity",
        "people_occupant",
        "co2_emission",
        "HVAC_electricity_demand_rate",
        "total_electricity_HVAC",
    ]

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
    for col in df.columns:
        print(f"\n{col}:")
        print(df[col].describe())

    # Check for potential temporal inconsistencies
    print("\nChecking for temporal consistency...")

    # Check month values
    if df["month"].min() < 1 or df["month"].max() > 12:
        print(
            f"Warning: Month values out of range: min={df['month'].min()}, max={df['month'].max()}"
        )
    else:
        print(
            f"Month values are in correct range: min={df['month'].min()}, max={df['month'].max()}"
        )

    # Check day values
    if df["day_of_month"].min() < 1 or df["day_of_month"].max() > 31:
        print(
            f"Warning: Day values out of range: min={df['day_of_month'].min()}, max={df['day_of_month'].max()}"
        )
    else:
        print(
            f"Day values are in correct range: min={df['day_of_month'].min()}, max={df['day_of_month'].max()}"
        )

    # Check hour values
    if df["hour"].min() < 0 or df["hour"].max() > 23:
        print(f"Warning: Hour values out of range: min={df['hour'].min()}, max={df['hour'].max()}")
    else:
        print(f"Hour values are in correct range: min={df['hour'].min()}, max={df['hour'].max()}")

    # Check for potential physical inconsistencies
    print("\nChecking for physical consistency...")

    # Check temperature ranges
    temp_cols = ["outdoor_temperature", "air_temperature"]
    for col in temp_cols:
        if col in df.columns:
            if df[col].min() < -50 or df[col].max() > 50:
                print(
                    f"Warning: {col} has extreme values: min={df[col].min()}, max={df[col].max()}"
                )
            else:
                print(f"{col} is within reasonable range: min={df[col].min()}, max={df[col].max()}")

    # Check humidity ranges
    humidity_cols = ["outdoor_humidity", "air_humidity"]
    for col in humidity_cols:
        if col in df.columns:
            if df[col].min() < 0 or df[col].max() > 100:
                print(
                    f"Warning: {col} has out-of-range values: min={df[col].min()}, max={df[col].max()}"
                )
            else:
                print(f"{col} is within valid range: min={df[col].min()}, max={df[col].max()}")

    # Check wind speed (shouldn't be negative)
    if "wind_speed" in df.columns:
        if df["wind_speed"].min() < 0:
            print(f"Warning: Negative wind speeds detected: min={df['wind_speed'].min()}")
        else:
            print(f"Wind speed values are non-negative: min={df['wind_speed'].min()}")

    # Check setpoint relationships
    if "htg_setpoint" in df.columns and "clg_setpoint" in df.columns:
        invalid_setpoints = df[df["htg_setpoint"] >= df["clg_setpoint"]]
        if not invalid_setpoints.empty:
            print(
                f"Warning: Found {len(invalid_setpoints)} rows where heating setpoint >= cooling setpoint"
            )
        else:
            print("All setpoints satisfy the constraint: heating < cooling")

    # Check for unrealistic electricity demand
    if "HVAC_electricity_demand_rate" in df.columns:
        if df["HVAC_electricity_demand_rate"].min() < 0:
            print(
                f"Warning: Negative electricity demand detected: min={df['HVAC_electricity_demand_rate'].min()}"
            )
        if df["HVAC_electricity_demand_rate"].max() > 1e6:  # Adjust threshold as needed
            print(
                f"Warning: Extremely high electricity demand detected: max={df['HVAC_electricity_demand_rate'].max()}"
            )
        print(
            f"HVAC electricity demand range: min={df['HVAC_electricity_demand_rate'].min()}, max={df['HVAC_electricity_demand_rate'].max()}"
        )

    # Create visualizations to help identify patterns and issues
    print("\nCreating visualizations...")

    # Create a subset of columns for visualization
    viz_cols = ["outdoor_temperature", "air_temperature", "HVAC_electricity_demand_rate"]
    viz_cols = [col for col in viz_cols if col in df.columns]

    if viz_cols:
        plt.figure(figsize=(15, 10))

        # Plot time series for selected columns
        for i, col in enumerate(viz_cols):
            plt.subplot(len(viz_cols), 1, i + 1)
            plt.plot(df[col])
            plt.title(f"{col} Time Series")
            plt.ylabel(col)
            if i == len(viz_cols) - 1:
                plt.xlabel("Time Step")
        plt.tight_layout()
        plt.savefig("observations_time_series.png")
        print("Time series visualizations saved to 'observations_time_series.png'")

        # Create correlation heatmap for numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr = df[num_cols].corr()
            plt.imshow(corr, cmap="coolwarm")
            plt.colorbar()
            plt.xticks(range(len(num_cols)), num_cols, rotation=90)
            plt.yticks(range(len(num_cols)), num_cols)
            plt.title("Correlation Heatmap")

            # Add correlation values
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    plt.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                    )
            plt.tight_layout()
            plt.savefig("observations_correlation.png")
            print("Correlation heatmap saved to 'observations_correlation.png'")
    print("\nValidation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate observations CSV file")
    parser.add_argument("file_path", type=str, help="Path to the observations CSV file")
    args = parser.parse_args()
    validate_observations(args.file_path)
