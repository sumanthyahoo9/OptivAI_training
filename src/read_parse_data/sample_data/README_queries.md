### We create different kinds of queries and some examples are given below
1. Predictive Queries
These ask the LLM to forecast setpoints, energy consumption, or temperature based on conditions.
Examples:

"Given outdoor temperature of 32°C, indoor temperature of 24.8°C, and 15 occupants at 2pm, what heating and cooling setpoints would you recommend?"
"If current indoor temperature is 23°C with 45% humidity and outdoor temperature is increasing from 28°C to 33°C over the next 3 hours, predict the optimal cooling setpoint trajectory."
"Based on historical data, what will the indoor temperature be in 2 hours if I maintain current setpoints of heating=20°C and cooling=24°C with outdoor temperature rising to 30°C?"
"Predict the HVAC electricity demand if I adjust cooling setpoint from 24°C to 22°C during peak occupancy on a hot day with outdoor temperature of 35°C."
"What setpoints should I use tomorrow between 9am-5pm if the weather forecast shows temperatures ranging from 18-26°C and I want to minimize energy use while maintaining comfort?"

2. Analytical Queries
These ask the LLM to explain relationships or analyze past performance.
Examples:

"Analyze the relationship between outdoor temperature and HVAC electricity demand in this building during summer months."
"Why does the agent set the cooling setpoint to 24.5°C rather than 23°C during periods of high occupancy in the afternoon?"
"Compare the energy efficiency of the HVAC system when outdoor temperatures are between 25-30°C versus 30-35°C."
"Explain the pattern of comfort violations in SPACE5-1 and their correlation with rapid changes in outdoor temperature."
"Analyze how the thermal mass of SPACE5-1 (volume 447.68 m³) affects temperature stability after setpoint changes."

3. Optimization Queries
These ask the LLM to suggest improvements to control strategies.
Examples:

"How could we modify the current control strategy to reduce energy consumption by 15% while keeping comfort violations below 5%?"
"What's the optimal pre-cooling strategy for SPACE5-1 during summer mornings to minimize afternoon peak demand?"
"Suggest an improved control policy for transitioning from unoccupied to occupied periods that minimizes energy spikes."
"If the comfort temperature range is widened from [20-25°C] to [19-26°C], how should the control strategy be adjusted to maximize energy savings?"
"Optimize the heating and cooling setpoints for a typical work week in July to balance comfort and energy use, given the zone's sizing parameters."

4. Diagnostic Queries
These ask the LLM to identify issues or anomalies in HVAC operation.
Examples:

"Identify periods where the HVAC system performed inefficiently, consuming more energy than expected for the given conditions."
"On July 15th, comfort violations increased despite normal setpoints. What factors might have contributed to this?"
"Review data from the past week and determine if any HVAC cycling or hunting behavior is occurring during low-load conditions."
"The indoor temperature frequently overshoots the cooling setpoint in the afternoons. What might be causing this behavior?"
"Analyze whether the current HVAC system is properly sized for SPACE5-1 based on its peak cooling and heating loads from the sizing data."

5. Contextual Understanding Queries
These ask the LLM to describe building behavior or summarize patterns.
Examples:

"Describe the typical thermal response of SPACE5-1 to a 2°C cooling setpoint decrease during occupied hours."
"Summarize the daily pattern of indoor temperature, setpoints, and energy consumption in SPACE5-1 during a typical summer weekday."
"How does SPACE5-1's thermal behavior differ between weekdays and weekends based on the observed data?"
"Explain how the building's thermal inertia affects temperature stability in SPACE5-1 during rapid outdoor temperature changes."
"Based on the data, characterize how occupancy patterns influence HVAC operation and energy consumption in SPACE5-1."

6. Policy Interpretation Queries
These ask the LLM to explain the reasoning behind control decisions.
Examples:

"At timestep 5240, the agent set cooling=24.8°C and heating=20.2°C. Explain the reasoning behind this decision given the conditions."
"Why does the control policy tend to raise cooling setpoints in the late afternoon despite continued occupancy?"
"The agent consistently uses wider deadbands between heating and cooling setpoints during nighttime. What's the rationale for this strategy?"
"Explain why the control strategy appears to prioritize maintaining a minimum temperature of 21.5°C even during periods of low occupancy."
"When outdoor temperatures exceed 30°C, what principles guide the agent's setpoint selection to balance comfort and energy use?"

These examples can serve as templates for generating fine-tuning data for your LLM. Each query type helps the model develop different capabilities for understanding and controlling HVAC systems, ultimately creating a more versatile assistant for building energy management.