"""
Use this script to test the fine-tuned model
"""
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class HVACModelEvaluator:
    """
    Evaluation class for the fine-tuned model
    """
    def __init__(self, model_path, tokenizer_path, test_data_path, reference_csv_path):
        """Initialize the evaluator with model, tokenizer, and data paths"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load test data
        self.test_data = []
        with open(test_data_path, 'r') as f:
            for line in f:
                self.test_data.append(json.loads(line))
        
        # Load reference CSV
        self.reference_csv = pd.read_csv(reference_csv_path)
        
        # Metrics storage
        self.results = {
            'response_quality': [],
            'temperature_accuracy': [],
            'setpoint_accuracy': [],
            'energy_recommendation_accuracy': [],
            'semantic_similarity': []
        }
    
    def extract_numerical_values(self, text):
        """Extract temperature and setpoint values from response text"""
        # Extract temperatures and setpoints using regex
        temp_pattern = r'(\d+\.?\d*)\s*°?C'
        setpoint_pattern = r'setpoint.*?(\d+\.?\d*)\s*°?C'
        
        temperatures = re.findall(temp_pattern, text)
        setpoints = re.findall(setpoint_pattern, text)
        
        return {
            'temperatures': [float(t) for t in temperatures],
            'setpoints': [float(s) for s in setpoints]
        }
    
    def evaluate_temperature_prediction(self, predicted_text, ground_truth_text):
        """Evaluate accuracy of temperature predictions"""
        pred_values = self.extract_numerical_values(predicted_text)
        gt_values = self.extract_numerical_values(ground_truth_text)
        
        if pred_values['temperatures'] and gt_values['temperatures']:
            # Calculate MSE and MAE for temperature predictions
            pred_temps = pred_values['temperatures'][:len(gt_values['temperatures'])]
            gt_temps = gt_values['temperatures'][:len(pred_values['temperatures'])]
            
            if pred_temps and gt_temps:
                mse = mean_squared_error(gt_temps, pred_temps)
                mae = mean_absolute_error(gt_temps, pred_temps)
                return {'mse': mse, 'mae': mae}
        
        return {'mse': float('inf'), 'mae': float('inf')}
    
    def evaluate_setpoint_recommendation(self, predicted_text, ground_truth_text):
        """Evaluate setpoint recommendation accuracy"""
        pred_values = self.extract_numerical_values(predicted_text)
        gt_values = self.extract_numerical_values(ground_truth_text)
        
        setpoint_accuracy = 0
        if pred_values['setpoints'] and gt_values['setpoints']:
            # Compare heating and cooling setpoints
            if 'heating' in predicted_text.lower() and 'heating' in ground_truth_text.lower():
                pred_heating = [s for i, s in enumerate(pred_values['setpoints']) if 'heating' in predicted_text.lower()[predicted_text.lower().find(str(s)):]]
                gt_heating = [s for i, s in enumerate(gt_values['setpoints']) if 'heating' in ground_truth_text.lower()[ground_truth_text.lower().find(str(s)):]]
                
                if pred_heating and gt_heating:
                    setpoint_accuracy += 1 - abs(pred_heating[0] - gt_heating[0]) / gt_heating[0]
        
        return setpoint_accuracy
    
    def evaluate_energy_recommendation(self, predicted_text, ground_truth_text):
        """Evaluate energy-related recommendations"""
        energy_keywords = ['energy', 'consumption', 'efficiency', 'power', 'demand']
        
        pred_energy_mentions = sum(1 for keyword in energy_keywords if keyword in predicted_text.lower())
        gt_energy_mentions = sum(1 for keyword in energy_keywords if keyword in ground_truth_text.lower())
        
        if gt_energy_mentions > 0:
            # Check if energy-related recommendations are aligned
            pred_increase = any(term in predicted_text.lower() for term in ['increase', 'raise', 'higher'])
            pred_decrease = any(term in predicted_text.lower() for term in ['decrease', 'lower', 'reduce'])
            
            gt_increase = any(term in ground_truth_text.lower() for term in ['increase', 'raise', 'higher'])
            gt_decrease = any(term in ground_truth_text.lower() for term in ['decrease', 'lower', 'reduce'])
            
            if (pred_increase and gt_increase) or (pred_decrease and gt_decrease):
                return 1.0
        
        return 0.0
    
    def get_csv_context(self, query):
        """Extract relevant CSV data based on query context"""
        # Extract timestamp from query if available
        time_pattern = r'(?:at|hour|time)\s*(\d+)'
        matches = re.findall(time_pattern, query)
        
        if matches:
            hour = int(matches[0])
            # Find data points near this hour
            relevant_data = self.reference_csv[self.reference_csv['hour'] == hour]
            if len(relevant_data) > 0:
                latest_data = relevant_data.iloc[-1]
                return {
                    'indoor_temp': latest_data['obs_air_temperature'],
                    'outdoor_temp': latest_data['obs_outdoor_temperature'],
                    'heating_setpoint': latest_data['action_Heating_Setpoint_RL'],
                    'cooling_setpoint': latest_data['action_Cooling_Setpoint_RL']
                }
        
        # Return latest data as fallback
        latest_data = self.reference_csv.iloc[-1]
        return {
            'indoor_temp': latest_data['obs_air_temperature'],
            'outdoor_temp': latest_data['obs_outdoor_temperature'],
            'heating_setpoint': latest_data['action_Heating_Setpoint_RL'],
            'cooling_setpoint': latest_data['action_Cooling_Setpoint_RL']
        }
    
    def generate_enriched_prompt(self, query):
        """Create an enriched prompt with CSV context"""
        csv_context = self.get_csv_context(query)
        
        enriched_prompt = f"""Current building conditions (SPACE5-1):
- Indoor temperature: {csv_context['indoor_temp']:.1f}°C
- Outdoor temperature: {csv_context['outdoor_temp']:.1f}°C
- Current heating setpoint: {csv_context['heating_setpoint']:.1f}°C
- Current cooling setpoint: {csv_context['cooling_setpoint']:.1f}°C

Question: {query}

Please provide a detailed response considering these current conditions."""
        
        return enriched_prompt
    
    def generate_response(self, query):
        """Generate response using the fine-tuned model with CSV context"""
        enriched_query = self.generate_enriched_prompt(query)
        
        # Format for chat model
        messages = [
            {"role": "system", "content": "You are an HVAC expert assistant."},
            {"role": "user", "content": enriched_query}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("assistant\n")[-1]
        
        return response
    
    def semantic_similarity(self, text1, text2):
        """Compute semantic similarity between two texts"""
        # Simple word overlap similarity (can be improved with embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def evaluate_single_example(self, example):
        """Evaluate a single test example"""
        query = example['query']
        ground_truth = example['response']
        
        # Generate prediction with CSV context
        prediction = self.generate_response(query)
        
        # Calculate metrics
        temp_metrics = self.evaluate_temperature_prediction(prediction, ground_truth)
        setpoint_acc = self.evaluate_setpoint_recommendation(prediction, ground_truth)
        energy_acc = self.evaluate_energy_recommendation(prediction, ground_truth)
        semantic_sim = self.semantic_similarity(prediction, ground_truth)
        
        return {
            'query': query,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'temperature_mse': temp_metrics['mse'],
            'temperature_mae': temp_metrics['mae'],
            'setpoint_accuracy': setpoint_acc,
            'energy_accuracy': energy_acc,
            'semantic_similarity': semantic_sim
        }
    
    def run_evaluation(self):
        """Run evaluation on all test examples"""
        print(f"Starting evaluation on {len(self.test_data)} examples...")
        
        for i, example in enumerate(self.test_data):
            print(f"Evaluating example {i+1}/{len(self.test_data)}")
            
            result = self.evaluate_single_example(example)
            
            # Store results
            self.results['temperature_accuracy'].append({
                'mse': result['temperature_mse'],
                'mae': result['temperature_mae']
            })
            self.results['setpoint_accuracy'].append(result['setpoint_accuracy'])
            self.results['energy_recommendation_accuracy'].append(result['energy_accuracy'])
            self.results['semantic_similarity'].append(result['semantic_similarity'])
            
            # Save detailed results
            self.save_detailed_result(i, result)
        
        return self.generate_report()
    
    def save_detailed_result(self, index, result):
        """Save detailed result for later analysis"""
        filename = f"evaluation_results/example_{index}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    def generate_report(self):
        """Generate final evaluation report"""
        report = {
            'overall_metrics': {},
            'summary': {}
        }
        
        # Calculate averages
        temp_mse = np.mean([r['mse'] for r in self.results['temperature_accuracy'] if r['mse'] != float('inf')])
        temp_mae = np.mean([r['mae'] for r in self.results['temperature_accuracy'] if r['mae'] != float('inf')])
        
        report['overall_metrics'] = {
            'average_temperature_mse': float(temp_mse),
            'average_temperature_mae': float(temp_mae),
            'average_setpoint_accuracy': float(np.mean(self.results['setpoint_accuracy'])),
            'average_energy_accuracy': float(np.mean(self.results['energy_recommendation_accuracy'])),
            'average_semantic_similarity': float(np.mean(self.results['semantic_similarity']))
        }
        
        # Generate visualizations
        self.create_visualizations()
        
        return report
    

    def create_visualizations(self):
        """Create visualization of evaluation metrics"""
        plt.figure(figsize=(15, 10))
        
        # Temperature accuracy distribution
        plt.subplot(2, 2, 1)
        valid_maes = [r['mae'] for r in self.results['temperature_accuracy'] if r['mae'] != float('inf')]
        if valid_maes:
            plt.hist(valid_maes, bins=20)
            plt.title('Temperature Prediction MAE Distribution')
            plt.xlabel('Mean Absolute Error (°C)')
            plt.ylabel('Frequency')
        
        # Setpoint accuracy distribution
        plt.subplot(2, 2, 2)
        plt.hist(self.results['setpoint_accuracy'], bins=20)
        plt.title('Setpoint Accuracy Distribution')
        plt.xlabel('Accuracy Score')
        plt.ylabel('Frequency')
        
        # Semantic similarity distribution
        plt.subplot(2, 2, 3)
        plt.hist(self.results['semantic_similarity'], bins=20)
        plt.title('Semantic Similarity Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        
        # Combined metrics
        plt.subplot(2, 2, 4)
        metrics = ['Temperature MAE', 'Setpoint Acc', 'Energy Acc', 'Semantic Sim']
        values = [
            np.mean([r['mae'] for r in self.results['temperature_accuracy'] if r['mae'] != float('inf')]),
            np.mean(self.results['setpoint_accuracy']),
            np.mean(self.results['energy_recommendation_accuracy']),
            np.mean(self.results['semantic_similarity'])
        ]
        plt.bar(metrics, values)
        plt.title('Overall Performance Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    


if __name__ == "__main__":
    evaluator = HVACModelEvaluator(
        model_path="./llama-3.1-8b-finetuned-hvac",
        tokenizer_path="./llama-3.1-8b-finetuned-hvac",
        test_data_path="test_data.jsonl",
        reference_csv_path="space5_training_data.csv"
    )
    
    # Run evaluation
    report = evaluator.run_evaluation()
    
    # Save report
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Evaluation complete. Check evaluation_report.json and evaluation_metrics.png for results.")