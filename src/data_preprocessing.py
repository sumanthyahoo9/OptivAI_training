"""
Runs a data preprocessing job and gets everything ready for the fine-tuning run.
"""
import json
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def extract_building_info(epjson_path, zones_data_path=None):
    """
    Extract building information from the epJSON file and zones data
    to create a comprehensive context
    """
    building_context = []
    
    # Load epJSON data
    try:
        with open(epjson_path, 'r') as f:
            data = json.load(f)
            
        # Extract basic building information
        if "Building" in data:
            building_props = list(data["Building"].values())[0]
            building_context.append("Building Information:")
            building_context.append(f"- North axis: {building_props.get('north_axis', 0)} degrees")
            building_context.append(f"- Terrain: {building_props.get('terrain', 'Not specified')}")
        
        # Extract location information
        if "Site:Location" in data:
            location_props = list(data["Site:Location"].values())[0]
            building_context.append("\nLocation Information:")
            building_context.append(f"- Location: {location_props.get('name', 'Not specified')}")
            building_context.append(f"- Latitude: {location_props.get('latitude', 'Not specified')}")
            building_context.append(f"- Longitude: {location_props.get('longitude', 'Not specified')}")
            building_context.append(f"- Time Zone: {location_props.get('time_zone', 'Not specified')}")
            building_context.append(f"- Elevation: {location_props.get('elevation', 'Not specified')} m")
        
        # Extract zone information
        if "Zone" in data:
            zones = data["Zone"]
            building_context.append("\nThermal Zones:")
            for zone_id, zone_props in zones.items():
                building_context.append(f"- {zone_id}:")
                building_context.append(f"  - Volume: {zone_props.get('volume', 'Not specified')} m³")
                building_context.append(f"  - Ceiling height: {zone_props.get('ceiling_height', 'Not specified')} m")
        
        # Extract HVAC system information
        if "AirLoopHVAC" in data:
            hvac_systems = data["AirLoopHVAC"]
            building_context.append(f"\nHVAC Systems:")
            for system_id, system_props in hvac_systems.items():
                building_context.append(f"- {system_id}")
    
    except Exception as e:
        print(f"Warning: Couldn't process epJSON file: {e}")
    
    # Load and add zones data if available
    if zones_data_path:
        try:
            # Add additional zones data
            building_context.append("\nAdditional Zone Details:")
            # Process zones data here
        except Exception as e:
            print(f"Warning: Couldn't process zones data: {e}")
    
    return "\n".join(building_context)

def prepare_fine_tuning_data(csv_path, observations_path, agent_actions_path, output_path, epjson_path=None):
    """
    Prepare fine-tuning dataset by combining query-response pairs with building context
    """
    # Load the main fine-tuning data
    df_training = pd.read_csv(csv_path)
    
    # Load observations and agent actions data for enriching context
    try:
        df_observations = pd.read_csv(observations_path)
        print(f"Loaded observations data with shape: {df_observations.shape}")
        
        df_actions = pd.read_csv(agent_actions_path)
        print(f"Loaded agent actions data with shape: {df_actions.shape}")
        
        # Extract building context if epJSON path is provided
        building_context = ""
        if epjson_path:
            building_context = extract_building_info(epjson_path)
            print(f"Extracted building context: {len(building_context.split('\n'))} lines")
        
        # Prepare the processed data
        processed_data = []
        
        for idx, row in tqdm(df_training.iterrows(), total=len(df_training), desc="Processing training data"):
            # Get the original query and response
            query = row['query']
            response = row['response']
            
            # Enrich with building context
            if building_context:
                # Add building context to the query
                enriched_query = f"Building Information Context:\n{building_context}\n\nUser Query: {query}"
            else:
                enriched_query = query
                
            # Add to processed data
            processed_data.append({
                "query": enriched_query,
                "response": response
            })
        
        # Create a DataFrame from the processed data
        df_processed = pd.DataFrame(processed_data)
        
        # Save the processed data
        df_processed.to_csv(output_path, index=False)
        print(f"Saved processed fine-tuning data to {output_path}")
        
        return df_processed
    
    except Exception as e:
        print(f"Error preparing fine-tuning data: {e}")
        return None

def create_synthetic_examples(observations_path, agent_actions_path, 
                              rewards_path, output_path, num_examples=100):
    """
    Create synthetic examples based on observations, agent actions, and rewards data
    """
    try:
        # Load the data
        df_observations = pd.read_csv(observations_path)
        df_actions = pd.read_csv(agent_actions_path)
        df_rewards = pd.read_csv(rewards_path)
        
        print("Creating synthetic examples based on actual HVAC control data...")
        
        # Select random samples (or you could choose based on specific criteria)
        indices = np.random.choice(len(df_observations)-1, num_examples, replace=False)
        
        synthetic_examples = []
        
        for idx in tqdm(indices, desc="Generating examples"):
            # Get observation data for the current timestep
            obs = df_observations.iloc[idx]
            
            # Get action data
            if idx < len(df_actions):
                action = df_actions.iloc[idx]
            else:
                continue
                
            # Get reward data
            if idx < len(df_rewards):
                reward = df_rewards.iloc[idx]['reward']
            else:
                continue
            
            # Create a query about the current state
            query_types = [
                f"The outdoor temperature is {obs['outdoor_temperature']:.1f}°C with {obs['outdoor_humidity']:.1f}% humidity. The indoor temperature is {obs['air_temperature']:.1f}°C. What should I set the heating and cooling setpoints to?",
                f"It's currently {obs['hour']:.0f}:00 and the outdoor temperature is {obs['outdoor_temperature']:.1f}°C. The HVAC system is consuming {obs['HVAC_electricity_demand_rate']:.1f} W. How can I optimize the setpoints?",
                f"Based on the current conditions: outdoor temp={obs['outdoor_temperature']:.1f}°C, indoor temp={obs['air_temperature']:.1f}°C, what are the recommended heating and cooling setpoints?",
                f"The current HVAC power demand is {obs['HVAC_electricity_demand_rate']:.1f} W and the indoor temperature is {obs['air_temperature']:.1f}°C. Suggest optimal temperature setpoints.",
                f"With {obs['people_occupant']:.0f} occupants and an indoor temperature of {obs['air_temperature']:.1f}°C, what heating and cooling setpoints would you recommend?"
            ]
            
            query = np.random.choice(query_types)
            
            # Create a response based on the actual action taken
            response = f"Based on the current conditions, I recommend setting the heating setpoint to {action['Heating_Setpoint_RL']:.1f}°C and the cooling setpoint to {action['Cooling_Setpoint_RL']:.1f}°C. This balances comfort and energy efficiency."
            
            # Add explanation based on reward
            if reward > 0:
                response += "This setting performed well in similar conditions, resulting in good energy efficiency while maintaining comfort."
            else:
                response += "This setting helps improve the performance from the current state, balancing comfort requirements with energy consumption."
            
            # Add some variety to the responses
            if np.random.random() < 0.3:
                response += f" Keep in mind that the outdoor temperature is {obs['outdoor_temperature']:.1f}°C, which influences the HVAC system's performance."
                
            if np.random.random() < 0.3 and obs['people_occupant'] > 0:
                response += f" With {obs['people_occupant']:.0f} occupants in the space, maintaining comfort is important."
                
            if np.random.random() < 0.3:
                response += "Monitor the system for any changes in conditions that might require further adjustments."
            
            synthetic_examples.append({
                "query": query,
                "response": response
            })
        
        # Create a DataFrame from the synthetic examples
        df_synthetic = pd.DataFrame(synthetic_examples)
        
        # Save the synthetic examples
        df_synthetic.to_csv(output_path, index=False)
        print(f"Saved {len(df_synthetic)} synthetic examples to {output_path}")
        
        return df_synthetic
    
    except Exception as e:
        print(f"Error creating synthetic examples: {e}")
        return None

def main():
    """
    The main function to run the data preprocessing
    """
    # Define paths properly
    observations_path = "observations.csv"
    agent_actions_path = "agent_actions.csv"
    rewards_path = "rewards.csv"
    epjson_path = "sample_epJSON.pdf"  # Use this as a placeholder for actual JSON content
    synthetic_examples_path = "synthetic_examples.csv"
    fine_tuning_data_path = "fine_tuning_data.csv"  # Original data
    processed_fine_tuning_data_path = "processed_fine_tuning_data.csv"  # Output path
    
    # 1. Create synthetic examples if needed
    create_synthetic_examples(
        observations_path=observations_path,
        agent_actions_path=agent_actions_path,
        rewards_path=rewards_path,
        output_path=synthetic_examples_path,
        num_examples=200
    )
    
    # 2. Process the main fine-tuning data
    prepare_fine_tuning_data(
        csv_path=fine_tuning_data_path,
        observations_path=observations_path,
        agent_actions_path=agent_actions_path,
        epjson_path=epjson_path,
        output_path=processed_fine_tuning_data_path
    )
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()