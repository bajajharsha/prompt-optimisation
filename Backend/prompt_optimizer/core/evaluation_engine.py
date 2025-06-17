"""
Evaluation Engine - Enhanced evaluation with configurable model support
"""

import sys
import os
from typing import Dict, List, Any
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.core.metric import JSONGenerationEvaluator

# Import model configuration types
try:
    import sys
    import os
    # Add fastapi_optimization_system to path if not already there
    fastapi_path = os.path.join(os.path.dirname(__file__), '..', '..', 'fastapi_optimization_system')
    if os.path.exists(fastapi_path) and fastapi_path not in sys.path:
        sys.path.insert(0, fastapi_path)
    
    from app.models.optimization_models import ModelConfiguration
    from app.services.model_service_factory import get_model_service_factory
    MODEL_SERVICE_AVAILABLE = True
    print("✅ Model service factory loaded successfully in evaluation_engine.py")
except ImportError as e:
    print(f"Warning: Model service factory not available in evaluation_engine, falling back to groq-only: {e}")
    MODEL_SERVICE_AVAILABLE = False
    
    # Fallback: try to import just groq service for backward compatibility
    try:
        from app.services.groq_service import get_groq_service
        GROQ_SERVICE_AVAILABLE = True
        print("✅ Groq service loaded successfully in evaluation_engine.py")
    except ImportError as e:
        print(f"Warning: Groq service not available in evaluation_engine, falling back to direct API calls: {e}")
        GROQ_SERVICE_AVAILABLE = False


class EvaluationEngine:
    """
    Enhanced evaluation engine that adds overall accuracy and handles different evaluation types
    Now supports configurable model providers
    """
    
    def __init__(self, model_config: ModelConfiguration = None):
        """
        Initialize evaluation engine with optional model configuration
        
        Args:
            model_config: Model configuration to use for evaluation. If None, defaults to groq
        """
        if model_config:
            self.model_config = model_config
        else:
            # No model configuration provided - this should not happen in production
            # Raise an error to force proper configuration
            raise ValueError(
                "Model configuration is required for EvaluationEngine. "
                "Please provide a ModelConfiguration with provider and model_name."
            )
        
        # Initialize model service factory if available
        if MODEL_SERVICE_AVAILABLE:
            self.model_service_factory = get_model_service_factory()
        else:
            self.model_service_factory = None
    
    async def evaluate_prompt(
        self,
        prompt: str,
        data: List[Dict[str, Any]],
        schema: Dict[str, List[str]],
        evaluation_type: str = "general",
        user_prompt_template: str = None,
        model_config: ModelConfiguration = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt on given data
        
        Args:
            prompt: The system prompt to evaluate (or combined prompt for backward compatibility)
            data: List of data samples with 'input' and 'expected_output'
            schema: JSON schema for validation
            evaluation_type: Type of evaluation for logging
            user_prompt_template: Optional user prompt template (if provided, prompt is treated as system prompt)
            model_config: Optional model configuration to override the default
            
        Returns:
            Enhanced metrics with overall accuracy
        """
        # Use provided model config or fall back to instance config
        active_model_config = model_config or self.model_config
        
        print(f"🔬 Running {evaluation_type} evaluation on {len(data)} samples...")
        try:
            provider = active_model_config.provider.value if hasattr(active_model_config.provider, 'value') else str(active_model_config.provider)
            model_name = active_model_config.model_name
            print(f"🤖 Using model: {provider}/{model_name}")
        except:
            print(f"🤖 Using model: groq/llama-3.3-70b-versatile (fallback)")
        
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
            predicted_texts = await self._run_inference_with_system_user_prompts(
                system_prompt=system_prompt,
                user_prompts=formatted_prompts,
                model_config=active_model_config
            )
        else:
            # Backward compatibility: combined prompt
            print("🤖 Running model inference with combined prompt...")
            predicted_texts = await self._run_batch_inference(
                prompts=input_prompts,
                base_prompt=prompt,
                model_config=active_model_config
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
        try:
            provider = active_model_config.provider.value if hasattr(active_model_config.provider, 'value') else str(active_model_config.provider)
            model_name = active_model_config.model_name
            model_used = f"{provider}/{model_name}"
        except:
            model_used = "groq/llama-3.3-70b-versatile"
        
        results['evaluation_metadata'] = {
            "evaluation_type": evaluation_type,
            "model_used": model_used,
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
                improvement = new_f1 - baseline_f1
                
                comparison["field_improvements"][field] = {
                    "f1_improvement": improvement,
                    "baseline_f1": baseline_f1,
                    "new_f1": new_f1
                }
        
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
    
    async def _run_batch_inference(
        self,
        prompts: List[str],
        base_prompt: str,
        model_config: ModelConfiguration
    ) -> List[str]:
        """
        Run batch inference using the configured model service
        
        Args:
            prompts: List of input prompts
            base_prompt: Base instruction prompt to use as system prompt
            model_config: Model configuration
            
        Returns:
            List of model responses
        """
        if MODEL_SERVICE_AVAILABLE and self.model_service_factory:
            try:
                return await self.model_service_factory.batch_inference(
                    model_config=model_config,
                    prompts=prompts,
                    system_prompt=base_prompt,
                    temperature=getattr(model_config, 'temperature', 0.2),
                    max_tokens=1024
                )
            except Exception as e:
                print(f"Error using model service factory, falling back: {e}")
        
        # Fallback to existing groq implementation for backward compatibility
        provider = model_config.provider.value.lower() if hasattr(model_config.provider, 'value') else str(model_config.provider).lower()
        if provider == "groq":
            from prompt_optimizer.core.metric import run_groq_inference
            return run_groq_inference(
                prompts=prompts,
                base_prompt=base_prompt,
                model=model_config.model_name
            )
        else:
            raise Exception(f"Model provider {provider} not supported in fallback mode")
    
    async def _run_inference_with_system_user_prompts(
        self,
        system_prompt: str,
        user_prompts: List[str],
        model_config: ModelConfiguration
    ) -> List[str]:
        """
        Run inference with separate system and user prompts
        
        Args:
            system_prompt: The system prompt (classification instructions)
            user_prompts: List of user prompts (queries to classify)
            model_config: Model configuration
            
        Returns:
            List of model responses
        """
        if MODEL_SERVICE_AVAILABLE and self.model_service_factory:
            try:
                return await self.model_service_factory.system_user_inference(
                    model_config=model_config,
                    system_prompt=system_prompt,
                    user_prompts=user_prompts,
                    temperature=getattr(model_config, 'temperature', 0.2),
                    max_tokens=1024
                )
            except Exception as e:
                print(f"Error using model service factory, falling back: {e}")
        
        # Fallback for groq only
        provider = model_config.provider.value.lower() if hasattr(model_config.provider, 'value') else str(model_config.provider).lower()
        if provider == "groq":
            # Use existing groq fallback from evaluation_engine.py
            if GROQ_SERVICE_AVAILABLE:
                try:
                    import asyncio
                    
                    async def _async_call():
                        try:
                            from app.services.groq_service import get_groq_service
                            groq_service = get_groq_service()
                            return await groq_service.inference_with_system_user_prompts(
                                system_prompt=system_prompt,
                                user_prompts=user_prompts,
                                model_name=model_config.model_name,
                                component="evaluation_engine",
                                operation="system_user_inference",
                                temperature=0.2,
                                max_completion_tokens=1024
                            )
                        except Exception as e:
                            print(f"Error in groq service call: {e}")
                            raise e
                    
                    return await _async_call()
                        
                except Exception as e:
                    print(f"Error using Groq service, falling back to direct API: {e}")
            
            # Direct API fallback for groq
            return self._fallback_groq_direct_api(system_prompt, user_prompts, model_config.model_name)
        else:
            raise Exception(f"Model provider {provider} not supported in fallback mode")
    
    def _fallback_groq_direct_api(
        self,
        system_prompt: str,
        user_prompts: List[str],
        model_name: str
    ) -> List[str]:
        """
        Fallback to direct Groq API calls
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
                    "model": model_name,
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
                        "model": model_name,
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/prompt_optimizer/core/evaluation_engine.py",
                        "component": "evaluation_engine_fallback",
                        "operation": "system_user_inference_direct"
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