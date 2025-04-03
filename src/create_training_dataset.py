"""
We use this script to collate the training data from different CSVs
observations.csv
simulated_actions.csv
agent_rewards.csv
epluszsz.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_file(file_path, prefix=None):
    """
    Load a CSV file and optionally add a prefix to column names

    Args:
        file_path (str): Path to the CSV file
        prefix (str, optional): Prefix to add to column names

    Returns:
        pandas.DataFrame or None: Loaded data or None if file not found
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return None

    try:
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)

        # Add prefix to column names if specified
        if prefix:
            rename_dict = {}
            for col in df.columns:
                # Skip time-related columns
                if col.lower() in ["time", "timestep", "month", "day_of_month", "hour"]:
                    rename_dict[col] = col
                else:
                    rename_dict[col] = f"{prefix}_{col}"
            df = df.rename(columns=rename_dict)

        print(f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def examine_epluszsz(file_path):
    """
    Examine the epluszsz.csv file to understand its structure and content

    Args:
        file_path (str): Path to the epluszsz.csv file

    Returns:
        dict: Information about the file structure
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return None

    try:
        df = pd.read_csv(file_path)

        # Check if there's a time column
        time_cols = [col for col in df.columns if col.lower() in ["time", "timestep"]]
        has_time_col = len(time_cols) > 0

        # Check for SPACE5-1 columns
        space5_cols = [col for col in df.columns if "SPACE5-1" in col]

        # Check column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Determine if the file contains daily data (96 rows ≈ 24 hours × 4 samples/hour)
        is_daily_data = df.shape[0] in [96, 97, 98]  # Allow slight variations

        info = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "has_time_column": has_time_col,
            "time_columns": time_cols,
            "space5_columns": space5_cols,
            "numeric_columns": numeric_cols,
            "is_daily_data": is_daily_data,
            "sample_data": df.head(2).to_dict() if df.shape[0] > 0 else None,
        }

        print(f"Examined epluszsz.csv: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Time columns: {time_cols}")
        print(f"SPACE5-1 columns: {len(space5_cols)}")
        print(f"Appears to be daily data: {is_daily_data}")

        return info
    except Exception as e:
        print(f"Error examining epluszsz.csv: {str(e)}")
        return None


def create_combined_dataset(output_path="space5_training_data.csv", include_epluszsz=True):
    """
    Create a combined dataset from all relevant files

    Args:
        output_path (str): Path to save the output CSV file
        include_epluszsz (bool): Whether to include data from epluszsz.csv

    Returns:
        pandas.DataFrame: The combined dataset
    """
    # Step 1: Load the action-related files
    rewards_df = load_file("src/read_parse_data/sample_data/rewards.csv", prefix="reward")
    agent_actions_df = load_file("src/read_parse_data/sample_data/agent_actions.csv", prefix="action")
    simulated_actions_df = load_file("src/read_parse_data/sample_data/simulated_actions.csv", prefix="simulated")

    # Step 2: Merge action-related data based on index (they all have 35040 rows)
    print("\nMerging action-related data...")

    # Add timestep column to each DataFrame based on index
    if rewards_df is not None:
        rewards_df["timestep"] = range(len(rewards_df))

    if agent_actions_df is not None:
        agent_actions_df["timestep"] = range(len(agent_actions_df))

    if simulated_actions_df is not None:
        simulated_actions_df["timestep"] = range(len(simulated_actions_df))

    # Start with rewards and merge others
    actions_combined = None

    if rewards_df is not None:
        actions_combined = rewards_df

        if agent_actions_df is not None:
            actions_combined = pd.merge(actions_combined, agent_actions_df, on="timestep", how="outer")

        if simulated_actions_df is not None:
            actions_combined = pd.merge(actions_combined, simulated_actions_df, on="timestep", how="outer")
    elif agent_actions_df is not None:
        actions_combined = agent_actions_df

        if simulated_actions_df is not None:
            actions_combined = pd.merge(actions_combined, simulated_actions_df, on="timestep", how="outer")
    elif simulated_actions_df is not None:
        actions_combined = simulated_actions_df

    # Add a dummy row at the beginning to represent timestep 0
    if actions_combined is not None:
        # Create a dummy row with NaN values
        dummy_row = pd.DataFrame({"timestep": [0]})
        for col in actions_combined.columns:
            if col != "timestep":
                dummy_row[col] = np.nan

        # Concatenate and reindex
        actions_combined = pd.concat([dummy_row, actions_combined]).reset_index(drop=True)
        print(f"Added dummy row at the beginning: Actions data now has {len(actions_combined)} rows")
    else:
        print("Warning: No action-related data found")
        return None

    # Step 3: Load observations data
    observations_df = load_file("src/read_parse_data/sample_data/observations.csv", prefix="obs")

    # Step 4: Merge observations with action-related data
    if observations_df is not None:
        print("\nMerging observations with action-related data...")

        # Add timestep column to observations based on index
        observations_df["timestep"] = range(len(observations_df))

        # Check if lengths match
        if len(observations_df) == len(actions_combined):
            print("Observations and actions data have matching lengths. Proceeding with merge.")
        else:
            print(f"Warning: Length mismatch - Observations: {len(observations_df)}, Actions: {len(actions_combined)}")
            print("Proceeding with merge based on timestep column.")

        # Merge based on timestep
        combined_df = pd.merge(observations_df, actions_combined, on="timestep", how="outer")

        print(f"Combined data after merging observations: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    else:
        print("Warning: Observations data not found. Using only action-related data.")
        combined_df = actions_combined

    # Step 5 (Optional): Examine and potentially incorporate epluszsz data
    if include_epluszsz:
        epluszsz_info = examine_epluszsz("epluszsz.csv")

        if epluszsz_info and epluszsz_info["is_daily_data"] and epluszsz_info["space5_columns"]:
            print("\nEvaluating how to incorporate epluszsz.csv data...")
            epluszsz_df = load_file("epluszsz.csv", prefix="size")

            # Since epluszsz contains daily data (96 rows), we need to determine how to integrate it
            # Options include:
            # 1. Create a separate dataset
            # 2. Extract relevant features to add to the combined dataset
            # 3. Repeat the daily pattern for the full year

            # For this script, we'll extract SPACE5-1 specific columns as a separate dataset
            space5_cols = epluszsz_info["space5_columns"]
            time_cols = epluszsz_info["time_columns"]

            if time_cols:
                space5_data = epluszsz_df[time_cols + space5_cols]
            else:
                # Create a time index if none exists
                epluszsz_df["hour_of_day"] = np.repeat(range(24), 4)
                epluszsz_df["quarter_hour"] = np.tile([0, 15, 30, 45], 24)
                space5_data = epluszsz_df[["hour_of_day", "quarter_hour"] + space5_cols]

            space5_data.to_csv("space5_sizing_data.csv", index=False)
            print(
                f"Saved SPACE5-1 sizing data to space5_sizing_data.csv:"
                f"{space5_data.shape[0]} rows, {space5_data.shape[1]} columns"
            )
            print("Note: This data represents daily patterns and was not merged with the time-series data.")

    # Step 6: Filter for SPACE5-1 specific data
    print("\nFiltering for SPACE5-1 specific data...")
    space5_cols = [col for col in combined_df.columns if "SPACE5-1" in col or "space5-1" in col.lower()]

    # Include general columns
    general_cols = [
        "timestep",
        "month",
        "day_of_month",
        "hour",
        "obs_outdoor_temperature",
        "obs_outdoor_humidity",
        "obs_wind_speed",
        "obs_wind_direction",
        "obs_diffuse_solar_radiation",
        "obs_direct_solar_radiation",
        "action_Heating_Setpoint_RL",
        "action_Cooling_Setpoint_RL",
        "reward_reward",
    ]
    general_cols = [col for col in general_cols if col in combined_df.columns]

    # If we found SPACE5-1 columns, filter the dataset
    if space5_cols:
        print(f"Found {len(space5_cols)} SPACE5-1 specific columns")
        selected_cols = general_cols + space5_cols

        # Include other relevant columns
        other_cols = [
            col
            for col in combined_df.columns
            if col not in selected_cols and not any(f"SPACE{i}" in col for i in range(1, 5)) and "zone" not in col.lower()
        ]

        final_cols = list(set(general_cols + space5_cols + other_cols))
        filtered_df = combined_df[final_cols]
        print(f"Selected {len(final_cols)} columns out of {len(combined_df.columns)}")
    else:
        print("No SPACE5-1 specific columns found. Using all columns.")
        filtered_df = combined_df

    # Step 7: Create a date column for better analysis
    if all(col in filtered_df.columns for col in ["month", "day_of_month", "hour"]):
        try:
            # Standardize column names
            month_col = [col for col in filtered_df.columns if "month" in col and "day" not in col][0]
            day_col = [col for col in filtered_df.columns if "day_of_month" in col][0]
            hour_col = [col for col in filtered_df.columns if "hour" in col and "quarter" not in col][0]

            # Create date column
            filtered_df["date"] = pd.to_datetime(
                filtered_df[[month_col, day_col]].assign(year=2021).astype(str).agg("-".join, axis=1)
            ) + pd.to_timedelta(filtered_df[hour_col], unit="h")

            # Move date column to the front
            cols = filtered_df.columns.tolist()
            cols.insert(1, "date")
            filtered_df = filtered_df[cols]
            print("Added date column for better temporal analysis")
        except Exception as e:
            print(f"Error creating date column: {str(e)}")

    # Step 8: Save the combined dataset
    print(f"\nSaving combined dataset to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(filtered_df)} rows and {len(filtered_df.columns)} columns of training data")

    # Generate a simple visualization to verify the data
    try:
        # Select some columns for visualization
        if "obs_outdoor_temperature" in filtered_df.columns and "reward_reward" in filtered_df.columns:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(filtered_df["obs_outdoor_temperature"].iloc[:1000])
            plt.title("Outdoor Temperature (First 1000 Data Points)")
            plt.ylabel("Temperature")

            plt.subplot(2, 1, 2)
            plt.plot(filtered_df["reward_reward"].iloc[:1000])
            plt.title("Reward (First 1000 Data Points)")
            plt.ylabel("Reward Value")
            plt.xlabel("Time Step")

            plt.tight_layout()
            plt.savefig("training_data_preview.png")
            print("Generated visualization of the training data (first 1000 points)")
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine data for SPACE5-1 zone from multiple CSV files")
    parser.add_argument("--output", type=str, default="space5_training_data.csv", help="Path to save the output CSV file")
    parser.add_argument("--skip-epluszsz", action="store_true", help="Skip processing epluszsz.csv")
    args = parser.parse_args()

    create_combined_dataset(args.output, not args.skip_epluszsz)
