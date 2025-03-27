"""
This script reads the agent's rewards and understands the data
"""
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validate_rewards(file_path):
    """
    Validate the rewards csv
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
    expected_columns = ["reward"]
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
    
    # Check for potential outliers using IQR method
    Q1 = df['reward'].quantile(0.25)
    Q3 = df['reward'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['reward'] < lower_bound) | (df['reward'] > upper_bound)]
    if not outliers.empty:
        print(f"\nFound {len(outliers)} potential reward outliers using IQR method")
        print(f"IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print("Sample outliers (first 5):")
        print(outliers.head())
    else:
        print("\nNo reward outliers detected using IQR method")
    
    # Check for extreme values (assuming rewards are typically in a reasonable range)
    # Adjust these thresholds based on your expected reward range
    extreme_min, extreme_max = -10.0, 10.0
    extreme_values = df[(df['reward'] < extreme_min) | (df['reward'] > extreme_max)]
    if not extreme_values.empty:
        print(f"\nFound {len(extreme_values)} extremely large or small rewards")
        print("Sample extreme values (first 5):")
        print(extreme_values.head())
    else:
        print(f"\nNo extreme rewards detected outside range [{extreme_min}, {extreme_max}]")
    
    # Check for NaN, Inf, -Inf values
    invalid_values = df[~df['reward'].apply(lambda x: np.isfinite(x))]
    if not invalid_values.empty:
        print(f"\nFound {len(invalid_values)} non-finite values (NaN, Inf, -Inf)")
        print(invalid_values.head())
    else:
        print("\nNo non-finite values detected")
    
    # Visualization to help identify patterns and potential issues
    plt.figure(figsize=(12, 8))
    
    # Time series plot
    plt.subplot(2, 1, 1)
    plt.plot(df['reward'])
    plt.title('Reward Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    
    # Histogram of rewards
    plt.subplot(2, 1, 2)
    plt.hist(df['reward'], bins=50, alpha=0.75)
    plt.title('Reward Distribution')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('reward_analysis.png')
    print("\nVisualizations saved to 'reward_analysis.png'")
    
    # Additional checks for reward consistency
    print("\nChecking for constant rewards...")
    if df['reward'].nunique() == 1:
        print(f"Warning: All rewards have the same value: {df['reward'].iloc[0]}")
    else:
        print(f"Number of unique reward values: {df['reward'].nunique()}")
    
    # Check for long sequences of identical rewards
    reward_runs = []
    current_reward = None
    current_run = 0
    
    for reward in df['reward']:
        if reward == current_reward:
            current_run += 1
        else:
            if current_run > 100:  # Arbitrary threshold for suspiciously long sequences
                reward_runs.append((current_reward, current_run))
            current_reward = reward
            current_run = 1
    
    # Check the last run
    if current_run > 100:
        reward_runs.append((current_reward, current_run))
    
    if reward_runs:
        print("\nFound suspicious long sequences of identical rewards:")
        for reward, length in reward_runs:
            print(f"Reward value {reward}: {length} consecutive occurrences")
    else:
        print("\nNo suspiciously long sequences of identical rewards detected")

# Run the validation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate CSV file')
    parser.add_argument('file_path', type=str, help='Path to the CSV file to validate')
    args = parser.parse_args()
    if args.file_path.endswith('rewards.csv'):
        validate_rewards(args.file_path)
    else:
        print(f"Unsupported file type: {args.file_path}")
        print("Supported files: agent_actions.csv, rewards.csv")
        sys.exit(1)