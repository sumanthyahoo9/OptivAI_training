"""
Use this script to create variations/diversity in the dataset used for fine-tuning
"""
import os
import random
import json
from tqdm import tqdm
from typing import List, Dict
import asyncio
import openai

class HVACVariationGenerator:
    def __init__(self, api_key: str):
        """
        Constructor
        """
        self.api_key = api_key
        openai.api_key = api_key
        
    def create_variation_prompt(self, base_example: Dict, variation_type: str) -> str:
        """Create prompt for generating variations"""
        prompts = {
            'style': """Generate 5 variations of this HVAC query-response pair, each with different writing styles:
1. Technical/Professional
2. Conversational/Friendly
3. Concise/Direct
4. Detailed/Educational
5. Problem-solving focused

Original:
Query: {query}
Response: {response}
Context: Indoor temp: {indoor_temp:.1f}°C, Outdoor temp: {outdoor_temp:.1f}°C, Power: {power:.0f}W

Maintain the same technical accuracy but vary the tone and presentation. Each variation should feel natural for an HVAC expert system.

Format your response as a JSON array with objects containing 'query' and 'response' fields.""",

            'complexity': """Generate 5 variations of this HVAC query-response pair at different complexity levels:
1. Basic (for building occupants)
2. Intermediate (for facility staff)
3. Advanced (for HVAC technicians)
4. Expert (for engineers)
5. Executive summary (for management)

Original:
Query: {query}
Response: {response}
Context: Indoor temp: {indoor_temp:.1f}°C, Outdoor temp: {outdoor_temp:.1f}°C, Power: {power:.0f}W

Adjust technical depth while maintaining accuracy. Format as JSON array.""",

            'perspective': """Generate 5 variations of this HVAC query-response pair from different perspectives:
1. Energy efficiency focused
2. Comfort optimization focused  
3. Cost reduction focused
4. Maintenance prevention focused
5. Sustainability focused

Original:
Query: {query}
Response: {response}
Context: Indoor temp: {indoor_temp:.1f}°C, Outdoor temp: {outdoor_temp:.1f}°C, Power: {power:.0f}W

Each should address the same issue but emphasize different aspects. Format as JSON array.""",

            'scenario': """Generate 5 variations of this HVAC query-response pair with different scenarios:
1. During a heatwave
2. During winter cold snap
3. With high occupancy
4. During off-hours
5. With equipment degradation

Original:
Query: {query}
Response: {response}
Context: Indoor temp: {indoor_temp:.1f}°C, Outdoor temp: {outdoor_temp:.1f}°C, Power: {power:.0f}W
Adapt the specifics while maintaining the core issue. Format as JSON array."""
        }
        
        context = base_example['context']
        prompt = prompts[variation_type].format(
            query=base_example['query'],
            response=base_example['response'],
            indoor_temp=context['conditions']['indoor_temp'],
            outdoor_temp=context['conditions']['outdoor_temp'],
            power=context['power']
        )
        
        return prompt
    
    async def generate_variations_async(self, base_example: Dict, variation_types: List[str]) -> List[Dict]:
        """Generate variations asynchronously"""
        variations = []
        
        for var_type in variation_types:
            prompt = self.create_variation_prompt(base_example, var_type)
            
            try:
                response = await self._call_openai_async(prompt)
                parsed_variations = self._parse_variations(response, base_example)
                variations.extend(parsed_variations)
            except Exception as e:
                print(f"Error generating {var_type} variations: {e}")
                continue
        
        return variations
    
    async def _call_openai_async(self, prompt: str) -> str:
        """Async OpenAI API call"""
        messages = [
            {"role": "system", "content": "You are an HVAC expert system that generates training data variations. Always respond with valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]
        
        # For GPT-4
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message['content']
    
    def _parse_variations(self, response: str, base_example: Dict) -> List[Dict]:
        """Parse API response and create variation examples"""
        variations = []
        
        try:
            # Extract JSON array from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            json_str = response[json_start:json_end]
            
            parsed = json.loads(json_str)
            
            for item in parsed:
                # Inherit context and metadata from base example
                variation = {
                    'query': item['query'],
                    'response': item['response'],
                    'context': base_example['context'],
                    'metadata': base_example['metadata'].copy()
                }
                variation['metadata']['is_variation'] = True
                variation['metadata']['base_example_id'] = id(base_example)
                
                variations.append(variation)
        except json.JSONDecodeError as e:
           print(f"Failed to parse JSON: {e}")
           print(f"Response: {response[:200]}...")
        except Exception as e:
           print(f"Error parsing variations: {e}")
       
        return variations
   
    async def generate_dataset_variations(self, base_examples: List[Dict], variations_per_example: int = 20) -> List[Dict]:
       """Generate variations for entire dataset"""
       all_variations = []
       variation_types = ['style', 'complexity', 'perspective', 'scenario']
       
       # Process in batches to avoid rate limits
       batch_size = 5
       
       for i in tqdm(range(0, len(base_examples), batch_size), desc="Generating variations"):
           batch = base_examples[i:i+batch_size]
           tasks = []
           
           for example in batch:
               # Generate variations of different types
               task = self.generate_variations_async(example, variation_types)
               tasks.append(task)
           
           # Wait for batch to complete
           batch_variations = await asyncio.gather(*tasks)
           
           for variations in batch_variations:
               all_variations.extend(variations)
           
           # Rate limiting
           await asyncio.sleep(1)
       
       return all_variations

# Additional utility functions for enhancing variations
class VariationEnhancer:
    def __init__(self):
        self.zone_names = ['SPACE1-1', 'SPACE2-1', 'SPACE3-1', 'SPACE4-1', 'SPACE5-1']
        self.time_contexts = {
            'morning': {'hours': range(6, 12), 'description': 'morning startup'},
            'afternoon': {'hours': range(12, 17), 'description': 'peak cooling hours'},
            'evening': {'hours': range(17, 21), 'description': 'evening wind-down'},
            'night': {'hours': range(21, 24), 'description': 'overnight conditions'}
        }
   
    def add_multi_turn_conversations(self, examples: List[Dict]) -> List[Dict]:
        """Convert single-turn examples into multi-turn conversations"""
        conversations = []
        
        # Group similar examples for multi-turn
        grouped = self._group_by_context(examples)
        
        for group in grouped:
            if len(group) >= 2:
                conversation = self._create_conversation(group)
                conversations.append(conversation)
        
        return conversations
   
    def _group_by_context(self, examples: List[Dict]) -> List[List[Dict]]:
        """Group examples by similar context"""
        groups = {}
        
        for example in examples:
            # Create a context key
            context = example['context']
            key = f"{context['conditions']['outdoor_temp']:.0f}_{context['occupancy']}_{example['metadata']['query_type']}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(example)
        
        return list(groups.values())
   
    def _create_conversation(self, examples: List[Dict]) -> Dict:
        """Create multi-turn conversation from related examples"""
        conversation = {
            'context': examples[0]['context'],
            'metadata': {
                'type': 'multi_turn',
                'turns': len(examples),
                'zone': examples[0]['metadata'].get('zone', 'SPACE5-1')
            },
            'conversation': []
        }
        
        for i, example in enumerate(examples[:3]):  # Limit to 3 turns
            conversation['conversation'].append({
                'turn': i + 1,
                'user': example['query'],
                'assistant': example['response']
            })
        
        return conversation
   
    def add_zone_diversity(self, examples: List[Dict]) -> List[Dict]:
        """Add zone diversity to examples"""
        enhanced = []
        
        for i, example in enumerate(examples):
            # Create copies for different zones
            if i % 5 == 0:  # Every 5th example, create zone variations
                for zone in self.zone_names:
                    zone_example = json.loads(json.dumps(example))  # Deep copy
                    zone_example['metadata']['zone'] = zone
                    # Update query and response to use new zone name
                    zone_example['query'] = zone_example['query'].replace('SPACE5-1', zone)
                    zone_example['response'] = zone_example['response'].replace('SPACE5-1', zone)
                    enhanced.append(zone_example)
            else:
                enhanced.append(example)
        
        return enhanced
   
    def add_error_handling_examples(self, examples: List[Dict]) -> List[Dict]:
        """Add examples that handle erroneous inputs"""
        error_examples = []
        
        error_templates = [
            {
                'query': "The heating setpoint is {invalid_heating}°C and cooling is {invalid_cooling}°C. Is this correct?",
                'response': "I notice there's an issue with your setpoint configuration. The heating setpoint ({invalid_heating}°C) is higher than the cooling setpoint ({invalid_cooling}°C), which creates a conflict. This can cause the system to cycle rapidly between heating and cooling, wasting energy.\n\nRecommended correction:\n• Heating setpoint: {corrected_heating}°C\n• Cooling setpoint: {corrected_cooling}°C\n\nThis maintains a proper deadband and prevents system conflicts.",
                'context_modifier': lambda ctx: {
                    'invalid_heating': 26.0,
                    'invalid_cooling': 24.0,
                    'corrected_heating': 22.0,
                    'corrected_cooling': 24.5
                }
            },
            {
                'query': "Why is the power showing {invalid_power}W? That seems impossible.",
                'response': "You're correct to question this reading. A power consumption of {invalid_power}W appears to be a sensor error or communication issue. Based on the current conditions:\n• Outdoor temp: {outdoor_temp:.1f}°C\n• Indoor temp: {indoor_temp:.1f}°C\n• Occupancy: {occupants}\n\nExpected power range: {min_power:.0f}W - {max_power:.0f}W\n\nRecommended actions:\n1. Check sensor connections\n2. Verify meter calibration\n3. Review system logs for communication errors",
                'context_modifier': lambda ctx: {
                    'invalid_power': 50000,
                    'outdoor_temp': ctx['conditions']['outdoor_temp'],
                    'indoor_temp': ctx['conditions']['indoor_temp'],
                    'occupants': ctx['occupancy'],
                    'min_power': 1000,
                    'max_power': 10000
                }
            }
        ]
        
        # Take 10% of examples to create error variants
        sample_size = max(1, len(examples) // 10)
        sampled = random.sample(examples, sample_size)
        
        for example in sampled:
            for error_template in error_templates:
                error_example = json.loads(json.dumps(example))  # Deep copy
                context_values = error_template['context_modifier'](example['context'])
                
                error_example['query'] = error_template['query'].format(**context_values)
                error_example['response'] = error_template['response'].format(**context_values)
                error_example['metadata']['type'] = 'error_handling'
                
                error_examples.append(error_example)
        
        return examples + error_examples

# Main generation script
async def generate_full_dataset(base_examples_path: str, output_path: str, api_key: str):
    """Generate full dataset with variations"""
    # Load base examples
    with open(base_examples_path, 'r') as f:
        base_examples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(base_examples)} base examples")
    
    # Generate variations
    generator = HVACVariationGenerator(api_key)
    variations = await generator.generate_dataset_variations(base_examples)
    
    print(f"Generated {len(variations)} variations")
    
    # Enhance variations
    enhancer = VariationEnhancer()
    
    # Add zone diversity
    enhanced = enhancer.add_zone_diversity(variations)
    
    # Add error handling examples
    enhanced = enhancer.add_error_handling_examples(enhanced)
    
    # Create multi-turn conversations
    conversations = enhancer.add_multi_turn_conversations(enhanced[:100])  # Use subset for conversations
    
    # Combine all examples
    all_examples = base_examples + enhanced + conversations
    
    print(f"Total examples: {len(all_examples)}")
    
    # Save to file
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    return all_examples

# Quality check function
def quality_check_examples(examples: List[Dict]) -> Dict:
    """Basic quality checks on generated examples"""
    stats = {
        'total': len(examples),
        'by_type': {},
        'avg_query_length': 0,
        'avg_response_length': 0,
        'has_numbers': 0,
        'has_context_reference': 0,
        'multi_turn': 0
    }
    
    query_lengths = []
    response_lengths = []
    
    for example in examples:
        # Type statistics
        metadata = example.get('metadata', {})
        query_type = metadata.get('query_type', 'unknown')
        stats['by_type'][query_type] = stats['by_type'].get(query_type, 0) + 1
        
        # Multi-turn
        if 'conversation' in example:
            stats['multi_turn'] += 1
            continue
        
        # Length statistics
        query = example.get('query', '')
        response = example.get('response', '')
        query_lengths.append(len(query))
        response_lengths.append(len(response))
        
        # Quality indicators
        if any(char.isdigit() for char in response):
            stats['has_numbers'] += 1
        
        if any(term in response.lower() for term in ['current', 'conditions', '°c', 'power', 'temperature']):
            stats['has_context_reference'] += 1
    
    stats['avg_query_length'] = sum(query_lengths) / len(query_lengths) if query_lengths else 0
    stats['avg_response_length'] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    return stats

# Usage example
if __name__ == "__main__":
    import os
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY")  # Set your API key
    base_examples_path = "hvac_base_examples.jsonl"
    output_path = "hvac_full_dataset.jsonl"
    
    # Run generation
    asyncio.run(generate_full_dataset(base_examples_path, output_path, api_key))
    
    # Quality check
    with open(output_path, 'r') as f:
        final_examples = [json.loads(line) for line in f]
    
    stats = quality_check_examples(final_examples)
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))