"""
Using OpenAI's API, this script generates query-response pairs to fine-tune an LLM
"""
import time
import json
import random
from typing import List, Dict, Any
import pandas as pd
import openai

def prepare_context_data(df: pd.DataFrame, row_idx: int, 
                        context_window: int = 24) -> Dict[str, Any]:
    """
    Extract relevant context from the dataset around a specific timestep.
    
    Args:
        df: The training dataframe
        row_idx: Index of the central row to focus on
        context_window: How many timesteps before/after to include (24 = 6 hours at 15-min intervals)
        
    Returns:
        Dictionary with contextual data
    """
    # Get window of data centered on the chosen index
    start_idx = max(0, row_idx - context_window)
    end_idx = min(len(df) - 1, row_idx + context_window)
    
    window_df = df.iloc[start_idx:end_idx + 1].copy()
    
    # Calculate some statistics from this window
    context = {
        "current_timestamp": {
            "month": int(df.iloc[row_idx]['month']),
            "day": int(df.iloc[row_idx]['day_of_month']),
            "hour": int(df.iloc[row_idx]['hour']),
            "indoor_temperature": round(float(df.iloc[row_idx]['obs_air_temperature']), 1),
            "outdoor_temperature": round(float(df.iloc[row_idx]['obs_outdoor_temperature']), 1),
            "humidity": round(float(df.iloc[row_idx]['obs_air_humidity']), 1),
            "occupants": int(df.iloc[row_idx]['obs_people_occupant']),
            "heating_setpoint": round(float(df.iloc[row_idx]['action_Heating_Setpoint_RL']), 1),
            "cooling_setpoint": round(float(df.iloc[row_idx]['action_Cooling_Setpoint_RL']), 1),
            "power_demand": round(float(df.iloc[row_idx]['obs_HVAC_electricity_demand_rate']), 1)
        },
        "window_statistics": {
            "avg_indoor_temp": round(window_df['obs_air_temperature'].mean(), 1),
            "min_indoor_temp": round(window_df['obs_air_temperature'].min(), 1),
            "max_indoor_temp": round(window_df['obs_air_temperature'].max(), 1),
            "avg_outdoor_temp": round(window_df['obs_outdoor_temperature'].mean(), 1),
            "min_outdoor_temp": round(window_df['obs_outdoor_temperature'].min(), 1),
            "max_outdoor_temp": round(window_df['obs_outdoor_temperature'].max(), 1),
            "avg_power_demand": round(window_df['obs_HVAC_electricity_demand_rate'].mean(), 1),
            "max_power_demand": round(window_df['obs_HVAC_electricity_demand_rate'].max(), 1)
        },
        "trends": {
            "outdoor_temp_trend": "rising" if window_df['obs_outdoor_temperature'].iloc[-1] > window_df['obs_outdoor_temperature'].iloc[0] else "falling",
            "indoor_temp_trend": "rising" if window_df['obs_air_temperature'].iloc[-1] > window_df['obs_air_temperature'].iloc[0] else "falling",
        },
        "metadata": {
            # Extract metadata columns if they exist
            "zone_volume": 447.68,  # From the previous epJSON analysis
            "design_cooling_load": df.iloc[row_idx].get('space5_1_sizing_des_sens_cool_load', 5000),
            "comfort_range": [20, 25]  # Default comfort range
        }
    }
    
    return context

def generate_llm_examples(
    training_data_path: str = "space5_training_data.csv", 
    num_examples: int = 500,
    output_file: str = "llm_enhanced_training_examples.jsonl",
    api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Generate high-quality training examples using an LLM
    
    Args:
        training_data_path: Path to the space5_training_data.csv
        num_examples: Total number of examples to generate
        output_file: Where to save the generated examples
        api_key: OpenAI API key
    """
    if api_key:
        openai.api_key = api_key
    
    # Load the training data
    df = pd.read_csv(training_data_path)
    print(f"Loaded training data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Define query types and their distribution
    query_types = {
        "predictive": 0.25,
        "analytical": 0.20,
        "optimization": 0.15,
        "diagnostic": 0.15,
        "contextual": 0.15,
        "policy": 0.10
    }
    
    # Calculate examples per type
    examples_per_type = {
        qtype: int(num_examples * ratio) for qtype, ratio in query_types.items()
    }
    
    # Ensure we have the right total by adjusting the largest category
    total = sum(examples_per_type.values())
    if total < num_examples:
        largest_type = max(examples_per_type, key=examples_per_type.get)
        examples_per_type[largest_type] += (num_examples - total)
    
    examples = []
    
    # For each query type, generate examples
    for query_type, count in examples_per_type.items():
        print(f"Generating {count} examples for query type: {query_type}")
        
        for i in range(count):
            # Randomly select a row from the dataframe
            # For better diversity, we could use stratified sampling by month/time of day
            idx = random.randint(0, len(df) - 1)
            
            # Prepare contextual data
            context = prepare_context_data(df, idx)
            
            # Construct prompt for the LLM
            prompt = f"""
You are an expert in HVAC systems and building energy management. 
I need you to generate a realistic, specific question about HVAC control for a building zone called SPACE5-1.

The question should be of type: {query_type.upper()}.

Here is the current context data:
- Month: {context['current_timestamp']['month']}
- Day: {context['current_timestamp']['day']}
- Hour: {context['current_timestamp']['hour']}
- Current indoor temperature: {context['current_timestamp']['indoor_temperature']}°C
- Current outdoor temperature: {context['current_timestamp']['outdoor_temperature']}°C
- Current humidity: {context['current_timestamp']['humidity']}%
- Current occupants: {context['current_timestamp']['occupants']}
- Current heating setpoint: {context['current_timestamp']['heating_setpoint']}°C
- Current cooling setpoint: {context['current_timestamp']['cooling_setpoint']}°C
- Current power demand: {context['current_timestamp']['power_demand']} W

Context statistics (last 6 hours):
- Average indoor temperature: {context['window_statistics']['avg_indoor_temp']}°C
- Min/Max indoor temperature: {context['window_statistics']['min_indoor_temp']}-{context['window_statistics']['max_indoor_temp']}°C
- Average outdoor temperature: {context['window_statistics']['avg_outdoor_temp']}°C
- Min/Max outdoor temperature: {context['window_statistics']['min_outdoor_temp']}-{context['window_statistics']['max_outdoor_temp']}°C
- Average power demand: {context['window_statistics']['avg_power_demand']} W
- Max power demand: {context['window_statistics']['max_power_demand']} W

Trends:
- Outdoor temperature is {context['trends']['outdoor_temp_trend']}
- Indoor temperature is {context['trends']['indoor_temp_trend']}

Building metadata:
- Zone volume: {context['metadata']['zone_volume']} m³
- Design cooling load: {context['metadata']['design_cooling_load']} W
- Comfort temperature range: {context['metadata']['comfort_range'][0]}-{context['metadata']['comfort_range'][1]}°C

Guidelines based on question type:
- PREDICTIVE: Ask about forecasting setpoints, energy consumption, or temperature based on conditions.
- ANALYTICAL: Ask about explaining relationships or analyzing past performance.
- OPTIMIZATION: Ask about improving control strategies for better performance.
- DIAGNOSTIC: Ask about identifying issues or anomalies in HVAC operation.
- CONTEXTUAL: Ask about describing building behavior or summarizing patterns.
- POLICY: Ask about explaining the reasoning behind control decisions.

Please generate only the question text. Make it specific, detailed, and directly related to the provided context.
"""
            
            try:
                # Call OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # or gpt-3.5-turbo
                    messages=[
                        {"role": "system", "content": "You are an HVAC expert assistant that generates realistic questions about building control systems."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                query = response.choices[0].message.content.strip()
                
                # Now generate a matching expert response
                answer_prompt = f"""
You are an expert in HVAC systems and building energy management.
Please provide a detailed, accurate answer to the following question about HVAC control for a building zone called SPACE5-1:

QUESTION: {query}

Here is the relevant context data:
- Month: {context['current_timestamp']['month']}
- Day: {context['current_timestamp']['day']}
- Hour: {context['current_timestamp']['hour']}
- Current indoor temperature: {context['current_timestamp']['indoor_temperature']}°C
- Current outdoor temperature: {context['current_timestamp']['outdoor_temperature']}°C
- Current humidity: {context['current_timestamp']['humidity']}%
- Current occupants: {context['current_timestamp']['occupants']}
- Current heating setpoint: {context['current_timestamp']['heating_setpoint']}°C
- Current cooling setpoint: {context['current_timestamp']['cooling_setpoint']}°C
- Current power demand: {context['current_timestamp']['power_demand']} W

Context statistics (last 6 hours):
- Average indoor temperature: {context['window_statistics']['avg_indoor_temp']}°C
- Min/Max indoor temperature: {context['window_statistics']['min_indoor_temp']}-{context['window_statistics']['max_indoor_temp']}°C
- Average outdoor temperature: {context['window_statistics']['avg_outdoor_temp']}°C
- Min/Max outdoor temperature: {context['window_statistics']['min_outdoor_temp']}-{context['window_statistics']['max_outdoor_temp']}°C
- Average power demand: {context['window_statistics']['avg_power_demand']} W
- Max power demand: {context['window_statistics']['max_power_demand']} W

Trends:
- Outdoor temperature is {context['trends']['outdoor_temp_trend']}
- Indoor temperature is {context['trends']['indoor_temp_trend']}

Building metadata:
- Zone volume: {context['metadata']['zone_volume']} m³
- Design cooling load: {context['metadata']['design_cooling_load']} W
- Comfort temperature range: {context['metadata']['comfort_range'][0]}-{context['metadata']['comfort_range'][1]}°C

Your answer should be technically accurate, detailed but concise, and directly address all aspects of the question.
"""
                
                answer_response = openai.ChatCompletion.create(
                    model="gpt-4",  # or gpt-3.5-turbo
                    messages=[
                        {"role": "system", "content": "You are an HVAC expert assistant that provides accurate, technical answers about building control systems."},
                        {"role": "user", "content": answer_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent answers
                    max_tokens=600
                )
                
                response = answer_response.choices[0].message.content.strip()
                
                # Add to examples
                example = {
                    "query": query,
                    "response": response,
                    "type": query_type,
                    "context": context,
                    "data_row_index": idx
                }
                examples.append(example)
                
                print(f"Generated example {i+1}/{count} for {query_type}")
                
                # Sleep to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating example: {e}")
                time.sleep(1)  # Wait longer on error
    
    # Save examples to a JSONL file
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(examples)} examples and saved to {output_file}")
    return examples

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LLM-enhanced training examples from HVAC data')
    parser.add_argument('--data', required=True, help='Path to space5_training_data.csv')
    parser.add_argument('--output', default='llm_enhanced_training_examples.jsonl', help='Output file path')
    parser.add_argument('--num-examples', type=int, default=500, help='Total number of examples to generate')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    
    args = parser.parse_args()
    
    examples = generate_llm_examples(
        args.data,
        num_examples=args.num_examples,
        output_file=args.output,
        api_key=args.api_key
    )