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

from poc.metric import JSONGenerationEvaluator, run_groq_inference


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
        evaluation_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt on given data
        
        Args:
            prompt: The prompt to evaluate
            data: List of data samples with 'input' and 'expected_output'
            schema: JSON schema for validation
            evaluation_type: Type of evaluation for logging
            
        Returns:
            Enhanced metrics with overall accuracy
        """
        print(f"🔬 Running {evaluation_type} evaluation on {len(data)} samples...")
        
        # Extract inputs and ground truth
        input_prompts = [sample['input'] for sample in data]
        ground_truth_jsons = [sample['expected_output'] for sample in data]
        
        # Run inference
        print("🤖 Running model inference...")
        predicted_texts = run_groq_inference(
            prompts=input_prompts,
            base_prompt=prompt,
            model=self.groq_model
        )
        
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
            "prompt_length": len(prompt)
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