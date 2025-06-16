"""
Evaluation Engine - Enhanced evaluation with overall accuracy
"""

import sys
import os
from typing import Dict, List, Any
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.core.metric import JSONGenerationEvaluator, run_groq_inference


class EvaluationEngine:
    """
    Enhanced evaluation engine that adds overall accuracy and handles different evaluation types
    """
    
    def __init__(self):
        self.groq_model = "llama-3.3-70b-versatile"
    
    async def evaluate_prompt(
        self,
        prompt: str,
        data: List[Dict[str, Any]],
        schema: Dict[str, List[str]],
        evaluation_type: str = "general",
        user_prompt_template: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt on given data
        
        Args:
            prompt: The system prompt to evaluate (or combined prompt for backward compatibility)
            data: List of data samples with 'input' and 'expected_output'
            schema: JSON schema for validation
            evaluation_type: Type of evaluation for logging
            user_prompt_template: Optional user prompt template (if provided, prompt is treated as system prompt)
            
        Returns:
            Enhanced metrics with overall accuracy
        """
        print(f"🔬 Running {evaluation_type} evaluation on {len(data)} samples...")
        
        # Extract inputs and ground truth
        input_prompts = [sample['input'] for sample in data]
        ground_truth_jsons = [sample['expected_output'] for sample in data]
        
        # Prepare prompts for inference
        if user_prompt_template:
            # New mode: separate system and user prompts
            system_prompt = prompt  # prompt is the system prompt
            # Combine user template with actual input data
            formatted_prompts = []
            for input_text in input_prompts:
                user_message = f"{user_prompt_template}\n\nInput to classify: {input_text}"
                formatted_prompts.append(user_message)
            
            print("🤖 Running model inference with separate system/user prompts...")
            predicted_texts = self._run_inference_with_system_user_prompts(
                system_prompt=system_prompt,
                user_prompts=formatted_prompts,
                model=self.groq_model
            )
        else:
            # Backward compatibility: combined prompt
            print("🤖 Running model inference with combined prompt...")
            predicted_texts = run_groq_inference(
                prompts=input_prompts,
                base_prompt=prompt,
                model=self.groq_model
            )
        print(f"Length of predicted texts: {len(predicted_texts)}")
        
        # Run evaluation using existing evaluator
        print("📊 Computing metrics...")
        evaluator = JSONGenerationEvaluator(schema)
        results = evaluator.evaluate_batch(
            ground_truth_jsons=ground_truth_jsons,
            predicted_texts=predicted_texts,
            input_prompts=input_prompts
        )
        
        # Add overall accuracy field
        overall_accuracy = self._calculate_overall_accuracy(results)
        results['overall_accuracy'] = overall_accuracy
        
        # Add evaluation metadata
        results['evaluation_metadata'] = {
            "evaluation_type": evaluation_type,
            "model_used": self.groq_model,
            "num_samples": len(data),
        }
        
        print(f"✅ {evaluation_type} evaluation completed:")
        print(f"   Overall Accuracy: {overall_accuracy:.3f}")
        print(f"   Average F1: {results['summary']['average_enum_macro_f1']:.3f}")
        print(f"   Valid JSON: {results['validation_metrics']['valid_json_accuracy']:.3f}")
        
        return results
    
    def _calculate_overall_accuracy(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall accuracy as correctly generated responses out of all
        
        Args:
            results: Results from JSONGenerationEvaluator
            
        Returns:
            Overall accuracy (0.0 to 1.0)
        """
        # Overall accuracy = exact match accuracy (correctly generated responses)
        return results['validation_metrics']['exact_match_accuracy']
    
    def compare_metrics(
        self,
        baseline_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare metrics between baseline and new results
        
        Args:
            baseline_metrics: Baseline evaluation results
            new_metrics: New evaluation results
            
        Returns:
            Comparison results with improvements/regressions
        """
        comparison = {
            "overall_accuracy_improvement": new_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy'],
            "f1_improvement": (
                new_metrics['summary']['average_enum_macro_f1'] - 
                baseline_metrics['summary']['average_enum_macro_f1']
            ),
            "valid_json_improvement": (
                new_metrics['validation_metrics']['valid_json_accuracy'] - 
                baseline_metrics['validation_metrics']['valid_json_accuracy']
            ),
            "failed_cases_change": (
                len(new_metrics['detailed_failed_cases']['wrong_classifications']) -
                len(baseline_metrics['detailed_failed_cases']['wrong_classifications'])
            )
        }
        
        # Field-level improvements
        field_improvements = {}
        for field in baseline_metrics['enum_field_metrics']:
            if field in new_metrics['enum_field_metrics']:
                baseline_f1 = baseline_metrics['enum_field_metrics'][field]['macro_f1']
                new_f1 = new_metrics['enum_field_metrics'][field]['macro_f1']
                field_improvements[field] = new_f1 - baseline_f1
        
        comparison['field_improvements'] = field_improvements
        
        # Overall assessment
        if comparison['overall_accuracy_improvement'] > 0.05:  # 5% improvement
            comparison['assessment'] = "SIGNIFICANT_IMPROVEMENT"
        elif comparison['overall_accuracy_improvement'] > 0.01:  # 1% improvement
            comparison['assessment'] = "MINOR_IMPROVEMENT"
        elif comparison['overall_accuracy_improvement'] > -0.01:  # Within 1%
            comparison['assessment'] = "NO_CHANGE"
        else:
            comparison['assessment'] = "REGRESSION"
        
        return comparison
    
    def get_comparison_fields(self) -> List[str]:
        """
        Get the fields that should be used for comparison
        
        Returns:
            List of field names for comparison
        """
        return [
            "overall_accuracy",
            "average_enum_macro_f1",
            "valid_json_accuracy",
            "failed_cases_count"
        ]
    
    def should_continue_optimization(
        self,
        baseline_metrics: Dict[str, Any],
        current_metrics: Dict[str, Any],
        improvement_threshold: float = 0.05
    ) -> bool:
        """
        Determine if optimization should continue based on Dev A metrics
        
        Args:
            baseline_metrics: Original baseline metrics
            current_metrics: Current iteration metrics
            improvement_threshold: Minimum improvement to continue
            
        Returns:
            True if optimization should continue
        """
        improvement = current_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy']
        return improvement >= improvement_threshold
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved to: {filepath}")
    
    def _run_inference_with_system_user_prompts(
        self,
        system_prompt: str,
        user_prompts: List[str],
        model: str
    ) -> List[str]:
        """
        Run inference with separate system and user prompts
        
        Args:
            system_prompt: The system prompt (classification instructions)
            user_prompts: List of user prompts (queries to classify)
            model: Model name to use
            
        Returns:
            List of model responses
        """
        import httpx
        import os
        from datetime import datetime
        from pymongo import MongoClient
        import pytz
        
        groq_api_key = os.getenv("groq_api_key")
        responses = []
        
        for user_prompt in user_prompts:
            try:
                # Setup the API request payload with separate system/user messages
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {groq_api_key}"
                }
                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    "model": model,
                    "temperature": 0.2,
                    "max_completion_tokens": 1024,
                    "stream": False,
                    "stop": None
                }
                
                # Make API request
                response = httpx.post(url, json=payload, headers=headers, verify=False)
                response_data = response.json()
                
                # Log tokens to MongoDB
                try:
                    client = MongoClient("mongodb://localhost:27017/")
                    db = client["personal_project_log_usage"]
                    collection = db["llm_usage"]
                    
                    usage = response_data.get("usage", {})
                    log_entry = {
                        "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                        "provider": "groq",
                        "model": model,
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/prompt_optimizer/core/evaluation_engine.py",
                        "component": "evaluation_engine",
                        "operation": "system_user_inference"
                    }
                    collection.insert_one(log_entry)
                except Exception as e:
                    print(f"Token logging failed: {e}")
                
                # Extract response content
                response_text = response_data["choices"][0]["message"]["content"]
                responses.append(response_text)
                
            except Exception as e:
                print(f"Error in Groq API call: {e}")
                responses.append("")  # Add empty string on error
        
        return responses 