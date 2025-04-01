"""
We use this script to collate the training data from different CSVs
observations.csv
simulated_actions.csv
agent_rewards.csv
epluszsz.csv
"""

import sys
import numpy as np
import pandas as pd

# Import the validation functions
try:
    from read_parse_data.read_agent_actions import validate_agent_actions
    from read_parse_data.read_observations import validate_observations
    from read_parse_data.read_agent_rewards import validate_rewards
    from read_parse_data.read_epluszsz import parse_epluszsz_data
except ImportError as e:
    print(f"Error importing validation functions: {e}")
    print("Ensure that the validation scripts are in the proper directory and return dataframes")
    sys.exit(1)


def create_space5_training_data(output_path="space5_training_data.csv"):
    """
    Create the training data for the zone SPACE5-1  using different CSVs

    Args:
        output_path (str, optional): _description_. Defaults to "space5_training_data.csv".
    """
    print("Loading and validating data from multiple sources...")

    # Load and validate data from each source
    print("\n=== Validating Observations Data ===")
    observations_df = validate_observations("sample_data/observations.csv")

    print("\n=== Validating Agent Actions Data ===")
    agent_actions_df = validate_agent_actions("sample_data/agent_actions.csv")

    print("\n=== Validating Rewards Data ===")
    rewards_df = validate_rewards("sample_data/rewards.csv")

    print("\n=== Extracting SPACE5-1 Data from epluszsz.csv ===")
    space5_sizing_df = parse_epluszsz_data("sample_data/epluszsz.csv")

    # Start with observations as the base dataset
    if observations_df is not None:
        print("\nUsing observations as the base dataset")
        combined_df = observations_df.copy()

        # Add timestep column if it doesn't exist
        if "hour" not in combined_df.columns:
            print("Adding timestep column to observations data")
            combined_df["hour"] = combined_df.index

        # Add agent actions
        if agent_actions_df is not None:
            print("\nMerging agent actions data")
            # Handle potential length mismatch
            if len(agent_actions_df) == len(combined_df) - 1:
                print("Length mismatch: Adding a dummy row to agent actions for alignment")
                # Add a dummy row for alignment
                agent_actions_df = pd.concat([agent_actions_df, pd.DataFrame([agent_actions_df.iloc[-1]])]).reset_index(
                    drop=True
                )

            # Add index as timestep if it doesn't exist
            if "hour" not in agent_actions_df.columns:
                agent_actions_df["hour"] = agent_actions_df.index

            # Merge with observations
            # The values have to be carefully aligned since observations has Time and the other scripts have Hour
            combined_df = pd.merge(combined_df, agent_actions_df, on="hour", how="left", suffixes=("", "_action"))
            print(f"Successfully merged agent actions data: {combined_df.shape}")

        # Add rewards
        if rewards_df is not None:
            print("\nMerging rewards data")
            # Handle potential length mismatch
            if "hour" not in rewards_df.columns:
                rewards_df["hour"] = rewards_df.index

            if len(rewards_df) == len(combined_df) - 1:
                print("Length mismatch: Adding a dummy row to rewards for alignment")
                # Add a dummy NaN row at the beginning for alignment
                dummy = pd.DataFrame({"hour": [0]})
                for col in rewards_df.columns:
                    if col != "hour":
                        dummy[col] = np.nan
                rewards_df = pd.concat([dummy, rewards_df]).reset_index(drop=True)
                rewards_df["hour"] = rewards_df.index

            # Merge with combined dataframe
            combined_df = pd.merge(combined_df, rewards_df, on="hour", how="left", suffixes=("", "_reward"))
            print(f"Successfully merged rewards data: {combined_df.shape}")

        # Add SPACE5-1 sizing data if available
        if space5_sizing_df is not None and not space5_sizing_df.empty:
            print("\nMerging SPACE5-1 sizing data")
            # Rename Time to timestep if it exists
            if "Time" in space5_sizing_df.columns:
                space5_sizing_df = space5_sizing_df.rename(columns={"Time": "timestep"})

            # Add index as timestep if it doesn't exist
            if "timestep" not in space5_sizing_df.columns:
                space5_sizing_df["timestep"] = space5_sizing_df.index

            # Merge with combined dataframe
            combined_df = pd.merge(combined_df, space5_sizing_df, on="timestep", how="left", suffixes=("", "_sizing"))
            print(f"Successfully merged SPACE5-1 sizing data: {combined_df.shape}")

        # Filter for SPACE5-1 related data and general environmental data
        print("\nFiltering for SPACE5-1 data and general environmental data")

        # Get columns specific to SPACE5-1
        space5_cols = [col for col in combined_df.columns if "SPACE5-1" in col or "space5-1" in col.lower()]
        print(f"Found {len(space5_cols)} columns specific to SPACE5-1")

        # Include general columns that aren't zone-specific
        general_cols = [
            "timestep",
            "month",
            "day_of_month",
            "hour",
            "outdoor_temperature",
            "outdoor_humidity",
            "wind_speed",
            "wind_direction",
            "diffuse_solar_radiation",
            "direct_solar_radiation",
            "Heating_Setpoint_RL",
            "Cooling_Setpoint_RL",
            "reward",
        ]
        general_cols = [col for col in general_cols if col in combined_df.columns]
        print(f"Found {len(general_cols)} general columns")

        # Combine general and SPACE5-1 specific columns
        selected_cols = general_cols + space5_cols

        # Include other relevant columns without zone specification
        other_cols = [
            col
            for col in combined_df.columns
            if col not in selected_cols and not any(f"SPACE{i}" in col for i in range(1, 5)) and "zone" not in col.lower()
        ]
        print(f"Found {len(other_cols)} other relevant columns")

        # Final column selection
        selected_cols = list(set(selected_cols + other_cols))

        # Create the final dataframe
        filtered_df = combined_df[selected_cols]
        print(f"\nSelected {len(selected_cols)} columns out of {len(combined_df.columns)}")

        # Check for any missing or duplicated data
        missing_count = filtered_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: Found {missing_count} missing values in the combined dataset")

        # Add a date column for better temporal analysis
        if all(col in filtered_df.columns for col in ["month", "day_of_month", "hour"]):
            filtered_df["date"] = pd.to_datetime(
                filtered_df[["month", "day_of_month"]].assign(year=2021).astype(str).agg("-".join, axis=1)
            ) + pd.to_timedelta(filtered_df["hour"], unit="h")
            print("Added date column for temporal analysis")

            # Move date column to the front
            cols = filtered_df.columns.tolist()
            cols.insert(1, cols.pop(cols.index("date")))
            filtered_df = filtered_df[cols]

        # Save to CSV
        print(f"\nSaving training data to {output_path}")
        filtered_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(filtered_df)} rows and {len(filtered_df.columns)} columns of training data")

        return filtered_df
    else:
        print("Error: Observations data is not available hence training data cannot be created.")
        return None
