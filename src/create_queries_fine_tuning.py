"""
Generate query-response pairs for fine-tuning
"""
import pandas as pd
import json
import random
from typing import Dict, List
import numpy as np

class HVACExampleGenerator:
    """
    Generate HVAC Examples
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.query_templates = self._initialize_query_templates()
        self.response_templates = self._initialize_response_templates()
        
    def _initialize_query_templates(self) -> Dict[str, List[str]]:
        """Initialize diverse query templates by category"""
        return {
            'diagnostic': [
                "Why is the power consumption {power:.0f}W when the outdoor temperature is {outdoor_temp:.1f}°C?",
                "The HVAC system is using {power:.0f}W. Is this normal for {occupants} occupants?",
                "What's causing the {comparison} power usage of {power:.0f}W in {zone}?",
                "The indoor temperature is {indoor_temp:.1f}°C but the setpoint is {setpoint:.1f}°C. What's wrong?",
                "Why isn't the system maintaining the {setpoint_type} setpoint of {setpoint:.1f}°C?",
            ],
            'predictive': [
                "Given the outdoor temperature of {outdoor_temp:.1f}°C {trend}, what's the expected power usage for the next {hours} hours?",
                "If the outdoor temperature continues {trend} from {outdoor_temp:.1f}°C, how will this affect energy consumption?",
                "What will happen to the indoor temperature if we maintain current setpoints?",
                "Based on {occupants} occupants and {outdoor_temp:.1f}°C outside, predict the cooling load.",
                "How will the {humidity:.0f}% humidity affect our power consumption in the next few hours?",
            ],
            'optimization': [
                "How can I reduce the {power:.0f}W power consumption while maintaining comfort?",
                "What setpoint adjustments would optimize energy use given {outdoor_temp:.1f}°C outside?",
                "The system is consuming {power:.0f}W. What changes would improve efficiency?",
                "How can we minimize energy use during {time_period} when electricity rates are {rate_level}?",
                "What's the most efficient temperature setpoint for {occupants} occupants?",
            ],
            'educational': [
                "Explain why outdoor temperature affects HVAC power consumption.",
                "How does humidity impact cooling efficiency?",
                "What's the relationship between occupancy and cooling load?",
                "Why does the system use more power during {time_period}?",
                "How do setpoint changes affect energy consumption?",
            ],
            'troubleshooting': [
                "The cooling setpoint is {clg_setpoint:.1f}°C but the room is at {indoor_temp:.1f}°C. What's the issue?",
                "Power consumption suddenly jumped to {power:.0f}W. What could be the cause?",
                "The system can't maintain temperature despite running at {power:.0f}W. Why?",
                "Indoor humidity is {humidity:.0f}% despite the AC running. What's wrong?",
                "The heating and cooling setpoints are {htg_setpoint:.1f}°C and {clg_setpoint:.1f}°C. Is this correct?",
            ],
        }
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize response structure templates"""
        return {
            'diagnostic': """Looking at the current conditions:

- Power consumption: {power:.0f}W {power_context}
- Outdoor temperature: {outdoor_temp:.1f}°C
- Indoor temperature: {indoor_temp:.1f}°C
- Humidity: {humidity:.0f}%
- Occupancy: {occupants} people

{analysis}

{cause}

{recommendation}""",
            
            'predictive': """Based on current conditions:

- Current power: {power:.0f}W
- Outdoor temp: {outdoor_temp:.1f}°C ({trend})
- Indoor temp: {indoor_temp:.1f}°C
- Setpoints: Heating {htg_setpoint:.1f}°C, Cooling {clg_setpoint:.1f}°C

{prediction}

{reasoning}

{confidence}""",
            
            'optimization': """Current system status:

- Power usage: {power:.0f}W
- Temperature differential: {temp_diff:.1f}°C (outdoor: {outdoor_temp:.1f}°C, indoor: {indoor_temp:.1f}°C)
- Current setpoints: Heating {htg_setpoint:.1f}°C, Cooling {clg_setpoint:.1f}°C

{current_efficiency}

Optimization recommendations:
{recommendations}

Expected results:
{expected_results}""",
        }
    
    def extract_context_from_row(self, row: pd.Series) -> Dict:
        """Extract structured context from CSV row"""
        return {
            'timestamp': {
                'month': int(row['month']),
                'day': int(row['day_of_month']),
                'hour': int(row['hour'])
            },
            'conditions': {
                'outdoor_temp': row['obs_outdoor_temperature'],
                'indoor_temp': row['obs_air_temperature'],
                'outdoor_humidity': row['obs_outdoor_humidity'],
                'indoor_humidity': row['obs_air_humidity'],
                'wind_speed': row['obs_wind_speed'],
                'solar_radiation': row['obs_direct_solar_radiation'] + row['obs_diffuse_solar_radiation']
            },
            'setpoints': {
                'heating': row['action_Heating_Setpoint_RL'] if pd.notna(row['action_Heating_Setpoint_RL']) else row['obs_htg_setpoint'],
                'cooling': row['action_Cooling_Setpoint_RL'] if pd.notna(row['action_Cooling_Setpoint_RL']) else row['obs_clg_setpoint']
            },
            'occupancy': int(row['obs_people_occupant']),
            'power': row['obs_HVAC_electricity_demand_rate'],
            'total_energy': row['obs_total_electricity_HVAC']
        }
    
    def determine_trends(self, current_idx: int, window: int = 6) -> Dict[str, str]:
        """Analyze trends in the data"""
        if current_idx < window:
            return {'outdoor_temp': 'stable', 'power': 'stable', 'indoor_temp': 'stable'}
        
        recent_data = self.df.iloc[current_idx-window:current_idx]
        
        trends = {}
        # Temperature trend
        temp_change = recent_data['obs_outdoor_temperature'].iloc[-1] - recent_data['obs_outdoor_temperature'].iloc[0]
        if temp_change > 1:
            trends['outdoor_temp'] = 'rising'
        elif temp_change < -1:
            trends['outdoor_temp'] = 'falling'
        else:
            trends['outdoor_temp'] = 'stable'
        
        # Power trend
        power_change = recent_data['obs_HVAC_electricity_demand_rate'].iloc[-1] - recent_data['obs_HVAC_electricity_demand_rate'].iloc[0]
        if power_change > 100:
            trends['power'] = 'increasing'
        elif power_change < -100:
            trends['power'] = 'decreasing'
        else:
            trends['power'] = 'stable'
            
        return trends
    
    def generate_analysis(self, context: Dict, query_type: str) -> str:
        """Generate contextual analysis based on conditions"""
        analyses = {
            'diagnostic': {
                'high_power': "The high power consumption of {power:.0f}W is primarily due to the significant temperature differential of {temp_diff:.1f}°C between indoor ({indoor_temp:.1f}°C) and outdoor ({outdoor_temp:.1f}°C) temperatures.",
                'low_power': "The relatively low power consumption of {power:.0f}W indicates efficient operation given the current {temp_diff:.1f}°C temperature differential.",
                'normal_power': "The current power consumption of {power:.0f}W is within expected parameters for these conditions.",
            },
            'optimization': {
                'high_outdoor_temp': "With outdoor temperatures at {outdoor_temp:.1f}°C, the cooling system is working hard to maintain comfort. Current efficiency is approximately {efficiency:.1f}W per degree of cooling.",
                'low_outdoor_temp': "The mild outdoor temperature of {outdoor_temp:.1f}°C presents an opportunity for free cooling or reduced mechanical cooling.",
                'high_occupancy': "With {occupants} occupants generating approximately {occupant_heat:.0f}W of internal heat, the cooling load is significantly increased.",
            }
        }
        
        power = context['power']
        temp_diff = abs(context['conditions']['outdoor_temp'] - context['conditions']['indoor_temp'])
        
        # Determine power level
        if power > 5000:
            power_level = 'high_power'
        elif power < 1000:
            power_level = 'low_power'
        else:
            power_level = 'normal_power'
        
        # Select appropriate analysis
        if query_type == 'diagnostic':
            analysis = analyses['diagnostic'][power_level]
        elif query_type == 'optimization':
            if context['conditions']['outdoor_temp'] > 30:
                analysis = analyses['optimization']['high_outdoor_temp']
            elif context['conditions']['outdoor_temp'] < 15:
                analysis = analyses['optimization']['low_outdoor_temp']
            else:
                analysis = analyses['optimization']['high_occupancy']
        else:
            analysis = f"System is operating with {power:.0f}W power consumption under current conditions."
        
        return analysis.format(
            power=power,
            temp_diff=temp_diff,
            indoor_temp=context['conditions']['indoor_temp'],
            outdoor_temp=context['conditions']['outdoor_temp'],
            occupants=context['occupancy'],
            occupant_heat=context['occupancy'] * 100,
            efficiency=power/temp_diff if temp_diff > 0 else 0
        )
    
    def create_base_example(self, row_idx: int, query_type: str) -> Dict:
        """Create a single high-quality base example"""
        row = self.df.iloc[row_idx]
        context = self.extract_context_from_row(row)
        trends = self.determine_trends(row_idx)
        
        # Select random templates
        query_template = random.choice(self.query_templates[query_type])
        
        # Prepare template variables
        template_vars = {
            'zone': 'SPACE5-1',
            'power': context['power'],
            'outdoor_temp': context['conditions']['outdoor_temp'],
            'indoor_temp': context['conditions']['indoor_temp'],
            'humidity': context['conditions']['indoor_humidity'],
            'occupants': context['occupancy'],
            'htg_setpoint': context['setpoints']['heating'],
            'clg_setpoint': context['setpoints']['cooling'],
            'setpoint': context['setpoints']['cooling'],  # Default to cooling
            'setpoint_type': 'cooling',
            'comparison': 'high' if context['power'] > 3000 else 'normal',
            'trend': trends['outdoor_temp'],
            'hours': random.choice([1, 2, 3, 4]),
            'time_period': self._get_time_period(row['hour']),
            'rate_level': 'high' if 9 <= row['hour'] <= 21 else 'low',
            'temp_diff': abs(context['conditions']['outdoor_temp'] - context['conditions']['indoor_temp'])
        }
        
        # Generate query
        query = query_template.format(**template_vars)
        
        # Generate response based on type
        response = self._generate_response(context, query_type, trends, template_vars)
        
        return {
            'query': query,
            'response': response,
            'context': context,
            'metadata': {
                'query_type': query_type,
                'timestamp': f"{int(row['month'])}-{int(row['day_of_month'])}-{int(row['hour'])}:00",
                'zone': 'SPACE5-1'
            }
        }
    
    def _generate_response(self, context: Dict, query_type: str, trends: Dict, template_vars: Dict) -> str:
        """Generate appropriate response based on query type"""
        if query_type == 'diagnostic':
            return self._generate_diagnostic_response(context, trends, template_vars)
        elif query_type == 'predictive':
            return self._generate_predictive_response(context, trends, template_vars)
        elif query_type == 'optimization':
            return self._generate_optimization_response(context, trends, template_vars)
        elif query_type == 'educational':
            return self._generate_educational_response(context, template_vars)
        else:  # troubleshooting
            return self._generate_troubleshooting_response(context, trends, template_vars)
    
    def _generate_diagnostic_response(self, context: Dict, trends: Dict, template_vars: Dict) -> str:
        """Generate diagnostic response"""
        analysis = self.generate_analysis(context, 'diagnostic')
        
        # Determine cause
        if context['power'] > 5000:
            cause = "The primary cause is the substantial cooling load created by:\n1. High outdoor temperature ({outdoor_temp:.1f}°C)\n2. {occupants} occupants adding internal heat\n3. Solar gain of {solar:.0f}W"
        else:
            cause = "The system is operating efficiently with minimal load due to:\n1. Moderate temperature differential\n2. Low occupancy ({occupants} people)\n3. Appropriate setpoint configuration"
        
        # Generate recommendation
        if context['power'] > 3000:
            recommendation = "To reduce power consumption:\n1. Increase cooling setpoint by 1-2°C\n2. Enable economizer if outdoor temp drops below {threshold:.1f}°C\n3. Check for unusual heat sources"
        else:
            recommendation = "Current operation is efficient. Maintain current setpoints and monitor for changes in conditions."
        
        response = self.response_templates['diagnostic'].format(
            power=context['power'],
            power_context='(above average)' if context['power'] > 3000 else '(within normal range)',
            outdoor_temp=context['conditions']['outdoor_temp'],
            indoor_temp=context['conditions']['indoor_temp'],
            humidity=context['conditions']['indoor_humidity'],
            occupants=context['occupancy'],
            analysis=analysis,
            cause=cause.format(**template_vars, solar=context['conditions']['solar_radiation']),
            recommendation=recommendation.format(threshold=context['conditions']['indoor_temp'] - 2)
        )
        
        return response
    
    def _generate_predictive_response(self, context: Dict, trends: Dict, template_vars: Dict) -> str:
        """Generate predictive response"""
        # Simple prediction model
        hours = template_vars['hours']
        current_power = context['power']
        
        if trends['outdoor_temp'] == 'rising':
            predicted_power = current_power * (1 + 0.05 * hours)
            prediction = f"Power consumption is expected to increase to approximately {predicted_power:.0f}W over the next {hours} hours."
        elif trends['outdoor_temp'] == 'falling':
            predicted_power = current_power * (1 - 0.03 * hours)
            prediction = f"Power consumption should decrease to approximately {predicted_power:.0f}W over the next {hours} hours."
        else:
            prediction = f"Power consumption will likely remain stable around {current_power:.0f}W."
        
        reasoning = "This prediction is based on:\n"
        reasoning += f"1. Outdoor temperature trend: {trends['outdoor_temp']}\n"
        reasoning += f"2. Current temperature differential: {template_vars['temp_diff']:.1f}°C\n"
        reasoning += f"3. Occupancy level: {context['occupancy']} people"
        
        confidence = "Confidence level: "
        if abs(predicted_power - current_power) < 500:
            confidence += "High (±10%)"
        else:
            confidence += "Moderate (±15%)"
        
        response = self.response_templates['predictive'].format(
            power=current_power,
            outdoor_temp=context['conditions']['outdoor_temp'],
            trend=trends['outdoor_temp'],
            indoor_temp=context['conditions']['indoor_temp'],
            htg_setpoint=context['setpoints']['heating'],
            clg_setpoint=context['setpoints']['cooling'],
            prediction=prediction,
            reasoning=reasoning,
            confidence=confidence
        )
        
        return response
    
    def _generate_optimization_response(self, context: Dict, trends: Dict, template_vars: Dict) -> str:
        """Generate optimization response"""
        current_efficiency = self.generate_analysis(context, 'optimization')
        
        recommendations = []
        expected_savings = 0
        
        # Temperature setpoint optimization
        if context['conditions']['outdoor_temp'] > 30:
            recommendations.append("1. Increase cooling setpoint from {clg:.1f}°C to {new_clg:.1f}°C".format(
                clg=context['setpoints']['cooling'],
                new_clg=context['setpoints']['cooling'] + 1
            ))
            expected_savings += 5
        
        # Occupancy-based optimization
        if context['occupancy'] == 0:
            recommendations.append("2. Switch to unoccupied mode with relaxed temperature limits")
            expected_savings += 15
        
        # Time-based optimization
        if template_vars['rate_level'] == 'high':
            recommendations.append("3. Pre-cool space before peak hours (9 AM - 9 PM)")
            recommendations.append("4. Increase setpoint by 2°C during peak hours")
            expected_savings += 10
        
        # Humidity optimization
        if context['conditions']['indoor_humidity'] > 60:
            recommendations.append("5. Optimize dehumidification settings")
            expected_savings += 5
        
        recommendations_text = '\n'.join(recommendations) if recommendations else "System is already well-optimized for current conditions."
        
        expected_results = f"• Energy savings: approximately {expected_savings}%\n"
        expected_results += f"• Reduced peak demand: {context['power'] * expected_savings / 100:.0f}W\n"
        expected_results += "• Maintained comfort within ±0.5°C of target"
        
        response = self.response_templates['optimization'].format(
            power=context['power'],
            temp_diff=template_vars['temp_diff'],
            outdoor_temp=context['conditions']['outdoor_temp'],
            indoor_temp=context['conditions']['indoor_temp'],
            htg_setpoint=context['setpoints']['heating'],
            clg_setpoint=context['setpoints']['cooling'],
            current_efficiency=current_efficiency,
            recommendations=recommendations_text,
            expected_results=expected_results
        )
        
        return response
    
    def _generate_educational_response(self, context: Dict, template_vars: Dict) -> str:
        """Generate educational response"""
        educational_content = {
            'temperature_effect': """The relationship between outdoor temperature and HVAC power consumption is nearly linear. For every 1°C increase in outdoor temperature above the setpoint, the cooling system typically uses 3-5% more energy.

In your current situation:
- Outdoor temp: {outdoor_temp:.1f}°C
- Indoor setpoint: {clg_setpoint:.1f}°C
- Temperature differential: {temp_diff:.1f}°C
- Resulting power: {power:.0f}W

This means your system is using approximately {power_per_degree:.0f}W per degree of cooling required.""",
            
            'humidity_impact': """Humidity significantly affects cooling efficiency through the latent heat load. When humidity is high, the HVAC system must remove moisture (dehumidify) in addition to cooling.

Current conditions:
- Indoor humidity: {humidity:.0f}%
- Latent load factor: approximately {latent_factor:.0f}% of total cooling

At {humidity:.0f}% humidity, your system uses about {extra_power:.0f}W extra compared to dry conditions.""",
            
            'occupancy_effect': """Each occupant generates approximately 100-120W of heat through metabolism and activity. This internal heat gain directly impacts cooling requirements.

With {occupants} occupants:
- Internal heat gain: {occupant_heat:.0f}W
- Additional cooling required: {cooling_for_occupants:.0f}W
- Impact on total load: {occupancy_percentage:.0f}%

This is why occupancy sensors and scheduling are valuable for energy optimization."""
        }
        
        # Select appropriate educational content
        topic = random.choice(list(educational_content.keys()))
        
        # Calculate educational metrics
        power_per_degree = context['power'] / template_vars['temp_diff'] if template_vars['temp_diff'] > 0 else 0
        latent_factor = min(30, context['conditions']['indoor_humidity'] / 2)
        extra_power = context['power'] * latent_factor / 100
        occupant_heat = context['occupancy'] * 100
        cooling_for_occupants = occupant_heat * 3.5  # COP assumption
        occupancy_percentage = (cooling_for_occupants / context['power'] * 100) if context['power'] > 0 else 0
        
        response = educational_content[topic].format(
            outdoor_temp=context['conditions']['outdoor_temp'],
            clg_setpoint=context['setpoints']['cooling'],
            temp_diff=template_vars['temp_diff'],
            power=context['power'],
            power_per_degree=power_per_degree,
            humidity=context['conditions']['indoor_humidity'],
            latent_factor=latent_factor,
            extra_power=extra_power,
            occupants=context['occupancy'],
            occupant_heat=occupant_heat,
            cooling_for_occupants=cooling_for_occupants,
            occupancy_percentage=min(100, occupancy_percentage)
        )
        
        return response
    
    def _generate_troubleshooting_response(self, context: Dict, trends: Dict, template_vars: Dict) -> str:
        """Generate troubleshooting response"""
        # Identify the issue
        setpoint_gap = context['conditions']['indoor_temp'] - context['setpoints']['cooling']
        
        if setpoint_gap > 2:
            issue = "The system cannot maintain the cooling setpoint"
            possible_causes = [
                "1. Insufficient cooling capacity for current load",
                "2. Refrigerant levels may be low",
                "3. Dirty filters restricting airflow",
                "4. Condenser coils need cleaning"
            ]
            solutions = [
                "1. Check and clean air filters",
                "2. Verify refrigerant charge",
                "3. Inspect condenser unit for blockages",
                "4. Consider increasing setpoint temporarily"
            ]
        elif context['power'] > 8000:
            issue = "Abnormally high power consumption"
            possible_causes = [
                "1. Compressor running continuously",
                "2. Refrigerant leak causing inefficiency",
                "3. Extreme temperature differential",
                "4. Control system malfunction"
            ]
            solutions = [
                "1. Check thermostat calibration",
                "2. Inspect for refrigerant leaks",
                "3. Verify control sequences",
                "4. Consider system maintenance"
            ]
        else:
            issue = "Minor temperature control deviation"
            possible_causes = [
                "1. Normal thermal lag",
                "2. Recent setpoint change",
                "3. Changing outdoor conditions"
            ]
            solutions = [
                "1. Allow system time to stabilize",
                "2. Monitor for persistent issues",
                "3. Check sensor calibration"
            ]
        
        response = f"""Troubleshooting Analysis for SPACE5-1:

Issue Identified: {issue}

Current Conditions:
- Indoor temperature: {context['conditions']['indoor_temp']:.1f}°C
- Cooling setpoint: {context['setpoints']['cooling']:.1f}°C
- Temperature deviation: {setpoint_gap:.1f}°C
- Power consumption: {context['power']:.0f}W
- System runtime: Continuous

Possible Causes:
{chr(10).join(possible_causes)}

Recommended Actions:
{chr(10).join(solutions)}

Priority: {'High' if setpoint_gap > 2 or context['power'] > 8000 else 'Medium'}

If issues persist after these checks, consider calling a qualified HVAC technician for detailed diagnostics."""
        
        return response
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period description"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def generate_base_examples(self, num_examples: int = 50) -> List[Dict]:
        """Generate specified number of base examples"""
        examples = []
        query_types = list(self.query_templates.keys())
        
        # Ensure diverse time coverage
        total_rows = len(self.df)
        indices = np.linspace(0, total_rows-1, num_examples, dtype=int)
        
        for i, idx in enumerate(indices):
            query_type = query_types[i % len(query_types)]
            example = self.create_base_example(idx, query_type)
            examples.append(example)
        
        return examples

# Generate base examples
def generate_base_dataset(csv_path: str, output_path: str, num_examples: int = 50):
    """Generate base dataset from CSV"""
    generator = HVACExampleGenerator(csv_path)
    examples = generator.generate_base_examples(num_examples)
    
    # Save examples
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(examples)} base examples")
    return examples

# Example usage
if __name__ == "__main__":
    csv_path = "space5_training_data.csv"
    output_path = "hvac_base_examples.jsonl"
    
    examples = generate_base_dataset(csv_path, output_path, num_examples=50)
    
    # Display a sample
    print("\nSample example:")
    print(json.dumps(examples[0], indent=2))