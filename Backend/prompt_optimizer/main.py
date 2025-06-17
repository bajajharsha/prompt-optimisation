#!/usr/bin/env python3
"""
Complete Prompt Optimization System
Combines all components into a working end-to-end system
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.core.data_manager import DataManager
from prompt_optimizer.core.evaluation_engine import EvaluationEngine
from prompt_optimizer.core.optimization_controller import OptimizationController
from prompt_optimizer.core.enhanced_human_feedback_fixed import create_simple_human_feedback_manager
from prompt_optimizer.core.request_id import initialize_request_id, get_request_id

class CompleteOptimizationSystem:
    """
    Main system that orchestrates the complete optimization workflow
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        # Don't initialize evaluation_engine here - will create when needed with model config
        self.evaluation_engine = None  
        self.optimization_controller = OptimizationController()
        self.enhanced_human_feedback = create_simple_human_feedback_manager()
        self.baseline_prompt = """You are a classification model. Classify the input into the correct category. Return the result in JSON format."""
        # Configuration
        self.config = {
            "max_iterations": 5,
            "improvement_threshold": 0.01,  # 1% improvement to continue
            "convergence_threshold": 0.005,  # Within 0.5% improvement considered same
            "convergence_patience": 3,  # Stop if stable for 3 iterations
            "max_retry_attempts": 2,  # Maximum retries when no improvement found
            "dataset_name": "code_gen",
            "target_model": {
                "provider": "groq",
                "model_name": "llama-3.3-70b-versatile"
            }
        }
        
        # Create model configuration object for EvaluationEngine
        self._model_config = self._create_model_config()
        
        # Schema definition
        self.schema = {
            "action": ["CODE_GENERATION", "NOT_FOUND"],
            "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
            "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", 
                        "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
            "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
            "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
        }
    
    def _create_model_config(self):
        """Create model configuration object"""
        try:
            # Import model configuration from fastapi system
            import sys
            import os
            fastapi_path = os.path.join(os.path.dirname(__file__), '..', 'fastapi_optimization_system')
            if os.path.exists(fastapi_path) and fastapi_path not in sys.path:
                sys.path.insert(0, fastapi_path)
            
            from app.models.optimization_models import ModelConfiguration, ModelProvider
            
            return ModelConfiguration(
                provider=ModelProvider.GROQ,
                model_name=self.config["target_model"]["model_name"],
                temperature=0.2,
                max_tokens=1024
            )
        except ImportError:
            # Fallback: return a simple object that has the required attributes
            class SimpleModelConfig:
                def __init__(self, model_name):
                    self.provider = type('Provider', (), {'value': 'groq'})()
                    self.model_name = model_name
                    self.temperature = 0.2
                    self.max_tokens = 1024
            
            return SimpleModelConfig(self.config["target_model"]["model_name"])
    
    def _get_evaluation_engine(self):
        """Get evaluation engine with proper model configuration"""
        if self.evaluation_engine is None:
            self.evaluation_engine = EvaluationEngine(model_config=self._model_config)
        return self.evaluation_engine
    
    async def run_complete_optimization(self) -> Dict[str, Any]:
        """
        Run the complete optimization workflow
        """
        # Initialize global request ID
        request_id = initialize_request_id()
        
        print("🚀 Starting Complete Prompt Optimization System")
        print("=" * 80)
        print(f"🆔 Request ID: {request_id}")
        
        try:
            # Step 1: Data Preparation
            print("\n📊 Step 1: Data Preparation and Stratification")
            data_splits = await self.data_manager.prepare_data_splits(
                dataset_name=self.config["dataset_name"]
            )
            
            print(f"✅ Data splits prepared:")
            print(f"   Train: {len(data_splits['train'])} samples (25%)")
            print(f"   Dev A: {len(data_splits['dev_a'])} samples (35%)")
            print(f"   Dev B: {len(data_splits['dev_b'])} samples (20%)")
            print(f"   Test: {len(data_splits['test'])} samples (20%)")
            
            # save data splits to folder with request id
            os.makedirs(f"fastapi_optimization_system/intermediate_results/{request_id}", exist_ok=True)
            with open(f"fastapi_optimization_system/intermediate_results/{request_id}/data_splits.json", "w") as f:
                json.dump(data_splits, f, indent=2, default=str)
            
            # Save baseline prompt
            await self._save_baseline_prompt(request_id)
                
            
            # Step 2: Train Baseline & Intent Analysis
            print("\n📈 Step 2: Train Baseline Evaluation")
            baseline_prompt = self._get_baseline_prompt()
            
            # Get baseline performance on TRAIN data for intent analysis
            train_baseline_metrics = await self._get_evaluation_engine().evaluate_prompt(
                prompt=baseline_prompt,
                data=data_splits['train'],
                schema=self.schema,
                evaluation_type="train_baseline"
            )
            
            # save train baseline metrics to folder with request id
            with open(f"fastapi_optimization_system/intermediate_results/{request_id}/train_baseline_metrics.json", "w") as f:
                json.dump(train_baseline_metrics, f, indent=2, default=str)
            
            print(f"✅ Train baseline evaluation completed:")
            print(f"   Overall Accuracy: {train_baseline_metrics['overall_accuracy']:.3f}")
            print(f"   Average F1: {train_baseline_metrics['summary']['average_enum_macro_f1']:.3f}")
            print(f"   Failed Cases: {len(train_baseline_metrics['detailed_failed_cases']['wrong_classifications'])}")
            
            # Step 3: Intent Analysis (based on train data)
            print("\n🎯 Step 3: Intent Analysis")
            intent_analysis = await self.optimization_controller.analyze_intent(
                schema=self.schema,
                baseline_metrics=train_baseline_metrics,  # Use train metrics for intent
                train_samples=data_splits['train'][:5],   # Use train samples for context
                base_prompt=baseline_prompt
            )
            
            # save intent analysis to folder with request id
            with open(f"fastapi_optimization_system/intermediate_results/{request_id}/intent_analysis.json", "w") as f:
                json.dump(intent_analysis, f, indent=2, default=str)
            
            print("✅ Intent analysis completed (based on train data)")
            
            # Step 4: Dev A Baseline (Hidden Target)
            print("\n📊 Step 4: Dev A Baseline (Optimization Target)")
            dev_a_baseline_metrics = await self._get_evaluation_engine().evaluate_prompt(
                prompt=baseline_prompt,
                data=data_splits['dev_a'],
                schema=self.schema,
                evaluation_type="dev_a_baseline"
            )
            
            # save dev a baseline metrics to folder with request id
            with open(f"fastapi_optimization_system/intermediate_results/{request_id}/dev_a_baseline_metrics.json", "w") as f:
                json.dump(dev_a_baseline_metrics, f, indent=2, default=str)
            
            print(f"✅ Dev A baseline evaluation completed:")
            print(f"   Overall Accuracy: {dev_a_baseline_metrics['overall_accuracy']:.3f}")
            print(f"   Average F1: {dev_a_baseline_metrics['summary']['average_enum_macro_f1']:.3f}")
            print(f"   Failed Cases: {len(dev_a_baseline_metrics['detailed_failed_cases']['wrong_classifications'])}")
            
            # Step 5: Optimization Loop
            print("\n🔄 Step 5: Optimization Loop")
            optimization_results = await self._run_optimization_loop(
                baseline_prompt=baseline_prompt,
                train_baseline_metrics=train_baseline_metrics,
                dev_a_baseline_metrics=dev_a_baseline_metrics,
                intent_analysis=intent_analysis,
                data_splits=data_splits
            )
            
            # save optimization results to folder with request id
            with open(f"fastapi_optimization_system/intermediate_results/{request_id}/optimization_results.json", "w") as f:
                json.dump(optimization_results, f, indent=2, default=str)
            
            # Step 6: Final Test Evaluation
            print("\n🧪 Step 6: Final Test Evaluation")
            final_results = await self._run_final_evaluation(
                optimization_results=optimization_results,
                train_baseline_metrics=train_baseline_metrics,
                dev_a_baseline_metrics=dev_a_baseline_metrics,
                test_data=data_splits['test']
            )
            
            print("\n✅ Complete optimization workflow finished!")
            return final_results
            
        except Exception as e:
            print(f"❌ Optimization workflow failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            await self.optimization_controller.close()
    
    async def _run_optimization_loop(
        self,
        baseline_prompt: str,
        train_baseline_metrics: Dict[str, Any],
        dev_a_baseline_metrics: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        data_splits: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Run the iterative optimization loop with improved stopping mechanism
        """
        current_prompt = baseline_prompt
        current_metrics = train_baseline_metrics
        iteration = 1
        optimization_history = []
        
        # Tracking for improved stopping mechanism
        improvement_history = []  # Track last improvements for convergence detection
        convergence_threshold = self.config["convergence_threshold"]  # Within 0.5% improvement considered same
        convergence_patience = self.config["convergence_patience"]  # Stop if stable for 3 iterations
        retry_attempts = 0
        max_retry_attempts = self.config["max_retry_attempts"]  # Maximum retries when no improvement found
        
        while iteration <= self.config["max_iterations"]:
            print(f"\n🔄 Optimization Iteration {iteration}")
            if retry_attempts > 0:
                print(f"   (Retry attempt {retry_attempts}/{max_retry_attempts})")
            print("-" * 50)
            
            # Generate candidate prompts with human feedback from previous iterations
            print("📝 Generating candidate prompts...")
            
            # Get human feedback from previous iteration if available
            previous_human_feedback = None
            if iteration > 1 and optimization_history:
                previous_iteration = optimization_history[-1]  # Get last iteration
                previous_human_feedback = previous_iteration.get('human_feedback_summary')
            
            candidates = await self.optimization_controller.generate_candidates(
                current_prompt=current_prompt,
                current_metrics=current_metrics,
                intent_analysis=intent_analysis,
                schema=self.schema,
                iteration=iteration,
                human_feedback_summary=previous_human_feedback,
                optimization_history=optimization_history
            )
            
            # Save generated candidate prompts to intermediate results
            if candidates:
                await self._save_candidate_prompts(candidates, iteration)
            
            if not candidates:
                print("❌ No candidates generated")
                if retry_attempts < max_retry_attempts:
                    retry_attempts += 1
                    print(f"🔄 Retrying optimization (attempt {retry_attempts}/{max_retry_attempts})...")
                    continue
                else:
                    print("❌ Maximum retry attempts reached, stopping optimization")
                    break
                        
            # Evaluate candidates on Dev A
            print("📊 Evaluating candidates on Dev A...")
            best_candidate = await self._evaluate_candidates_dev_a(
                candidates=candidates,
                dev_a_data=data_splits['dev_a'],
                baseline_metrics=current_metrics
            )
            
            if not best_candidate:
                print("❌ No improved candidates found")
                if retry_attempts < max_retry_attempts:
                    retry_attempts += 1
                    print(f"🔄 Retrying optimization (attempt {retry_attempts}/{max_retry_attempts})...")
                    continue
                else:
                    print("❌ Maximum retry attempts reached, stopping optimization")
                    break
            
            # Reset retry attempts on successful candidate generation
            retry_attempts = 0
            
            print(f"✅ Best candidate selected: {best_candidate['strategy']}")
            print(f"   Dev A Accuracy Improvement: {best_candidate['improvement']:.3f}")
            print(f"   Composite Score: {best_candidate['composite_score']:+.3f}")
            if best_candidate.get('f1_improvement', 0) != 0:
                print(f"   F1 Score Improvement: {best_candidate['f1_improvement']:+.3f}")
            if best_candidate.get('valid_json_improvement', 0) != 0:
                print(f"   Valid JSON Improvement: {best_candidate['valid_json_improvement']:+.3f}")
            if best_candidate.get('failed_cases_reduction', 0) != 0:
                print(f"   Failed Cases Reduction: {best_candidate['failed_cases_reduction']:+d}")
            
            # Save the selected best prompt to intermediate results
            await self._save_selected_prompt(best_candidate, iteration)
            
            # Add improvement to history for convergence tracking (use composite score for better tracking)
            composite_score = best_candidate.get('composite_score', best_candidate['improvement'])
            improvement_history.append(composite_score)
            
            # Check convergence: more flexible convergence detection
            if len(improvement_history) >= convergence_patience:
                recent_improvements = improvement_history[-convergence_patience:]
                min_improvement = min(recent_improvements)
                max_improvement = max(recent_improvements)
                improvement_range = max_improvement - min_improvement
                
                # Only consider convergence if we're in a good performance region
                recent_avg = sum(recent_improvements) / len(recent_improvements)
                is_performing_well = recent_avg > 0.01  # Average composite score is positive
                
                if improvement_range <= convergence_threshold and is_performing_well:
                    print(f"🎯 Optimization converged! Composite scores stable and performing well for {convergence_patience} iterations:")
                    for i, imp in enumerate(recent_improvements):
                        print(f"   Iteration {iteration - convergence_patience + i + 1}: {imp:.4f}")
                    print(f"   Range: {improvement_range:.4f} ≤ {convergence_threshold:.4f}")
                    print(f"   Average performance: {recent_avg:.4f}")
                    print("✅ Using current best prompt as final optimization result")
                    break
                elif improvement_range <= convergence_threshold and not is_performing_well:
                    print(f"⚠️  Scores stable but performance is low (avg: {recent_avg:.4f})")
                    print(f"   Continuing optimization to find better solutions...")
                elif iteration >= self.config["max_iterations"] - 1:
                    print(f"🔄 Reached maximum iterations ({self.config['max_iterations']})")
                    print(f"   Recent performance range: {improvement_range:.4f}")
                    print(f"   Will complete final iteration and stop")
            
            # Adaptive stopping criteria: consider multiple factors
            accuracy_improvement = best_candidate['improvement']
            composite_score = best_candidate.get('composite_score', accuracy_improvement)
            f1_improvement = best_candidate.get('f1_improvement', 0)
            failed_cases_reduction = best_candidate.get('failed_cases_reduction', 0)
            
            # Dynamic thresholds based on iteration and performance
            base_threshold = self.config["improvement_threshold"]
            
            # Make thresholds more lenient as iterations progress (exploration vs exploitation)
            iteration_factor = max(0.3, 1.0 - (iteration * 0.1))  # Gets more lenient over time
            adaptive_accuracy_threshold = base_threshold * iteration_factor
            adaptive_composite_threshold = base_threshold * iteration_factor * 0.3  # Even more lenient
            
            # Check for any meaningful improvement
            has_meaningful_improvement = (
                accuracy_improvement >= adaptive_accuracy_threshold or
                composite_score >= adaptive_composite_threshold or
                f1_improvement >= 0.02 or  # 2% F1 improvement is meaningful
                failed_cases_reduction >= 2 or  # Reducing 2+ failed cases is meaningful
                (accuracy_improvement > -0.01 and composite_score > 0)  # Small regression but positive composite
            )
            
            if has_meaningful_improvement:
                improvement_reasons = []
                if accuracy_improvement >= adaptive_accuracy_threshold:
                    improvement_reasons.append(f"accuracy improvement ({accuracy_improvement:.3f} ≥ {adaptive_accuracy_threshold:.3f})")
                if composite_score >= adaptive_composite_threshold:
                    improvement_reasons.append(f"composite score ({composite_score:.3f} ≥ {adaptive_composite_threshold:.3f})")
                if f1_improvement >= 0.02:
                    improvement_reasons.append(f"F1 improvement ({f1_improvement:.3f} ≥ 0.02)")
                if failed_cases_reduction >= 2:
                    improvement_reasons.append(f"failed cases reduction ({failed_cases_reduction} ≥ 2)")
                if accuracy_improvement > -0.01 and composite_score > 0:
                    improvement_reasons.append(f"stable accuracy with positive composite score")
                
                print(f"✅ Continuing optimization - meaningful improvement detected:")
                for reason in improvement_reasons:
                    print(f"   • {reason}")
            else:
                # Only stop if we've tried multiple iterations and see no improvement
                if iteration >= 2:  # Give at least 2 iterations before considering stopping
                    print(f"⏹️  No meaningful improvement detected after {iteration} iterations:")
                    print(f"     Accuracy improvement: {accuracy_improvement:.3f} (threshold: {adaptive_accuracy_threshold:.3f})")
                    print(f"     Composite score: {composite_score:.3f} (threshold: {adaptive_composite_threshold:.3f})")
                    print(f"     F1 improvement: {f1_improvement:.3f} (threshold: 0.02)")
                    print(f"     Failed cases reduction: {failed_cases_reduction} (threshold: 2)")
                    print("⏹️  Stopping optimization")
                    break
                else:
                    print(f"⚠️  Limited improvement in iteration {iteration}, but continuing to explore...")
                    print(f"     Will reassess after iteration {iteration + 1}")
            
            # Evaluate on Dev B for human feedback
            print("🔬 Evaluating on Dev B...")
            dev_b_results = await self._get_evaluation_engine().evaluate_prompt(
                prompt=best_candidate['optimized_prompt'],
                data=data_splits['dev_b'],
                schema=self.schema,
                evaluation_type="dev_b_validation"
            )
            
            print(f"✅ Dev B evaluation completed:")
            print(f"   Overall Accuracy: {dev_b_results['overall_accuracy']:.3f}")
            print(f"   Average F1: {dev_b_results['summary']['average_enum_macro_f1']:.3f}")
            
            # Collect enhanced human feedback with waiting mechanism
            print("👥 Collecting enhanced human feedback...")
            feedback_summary, human_feedback_results = await self.enhanced_human_feedback.collect_human_feedback_complete_workflow(
                candidate_prompt=best_candidate['optimized_prompt'],
                dev_b_results=dev_b_results,
                iteration=iteration,
                baseline_metrics=dev_a_baseline_metrics  # Use dev A baseline for comparison
            )
            
            # Update for next iteration
            current_prompt = best_candidate['optimized_prompt']
            current_metrics = dev_b_results
            
            # Store iteration results with enhanced feedback
            iteration_result = {
                "iteration": iteration,
                "strategy": best_candidate['strategy'],
                "dev_a_improvement": best_candidate['improvement'],
                "dev_b_metrics": dev_b_results,
                "human_feedback_summary": feedback_summary,  # Summarized feedback for context
                "human_feedback_details": human_feedback_results,  # Detailed results for analysis
                "optimized_prompt": best_candidate['optimized_prompt']
            }
            optimization_history.append(iteration_result)
            
            # Print human feedback summary for transparency
            if feedback_summary.total_cases > 0:
                print(f"📊 Human Feedback Summary:")
                print(f"   Correct: {feedback_summary.correct_count}, Incorrect: {feedback_summary.incorrect_count}, Skipped: {feedback_summary.skipped_count}")
                print(f"   Reviewer confidence: {feedback_summary.average_confidence:.2f}")
                if feedback_summary.key_feedback_themes:
                    print(f"   Key themes: {', '.join(feedback_summary.key_feedback_themes[:3])}")
                if feedback_summary.improvement_suggestions:
                    print(f"   Suggestions: {feedback_summary.improvement_suggestions[0][:50]}...")
            
            print(f"✅ Iteration {iteration} completed")
            iteration += 1
        
        # Add stopping reason to results
        stopping_reason = "max_iterations_reached"
        if len(improvement_history) >= convergence_patience:
            recent_improvements = improvement_history[-convergence_patience:]
            min_improvement = min(recent_improvements)
            max_improvement = max(recent_improvements)
            if max_improvement - min_improvement <= convergence_threshold:
                stopping_reason = "converged"
        elif improvement_history and improvement_history[-1] < self.config["improvement_threshold"]:
            stopping_reason = "below_threshold"
        elif retry_attempts >= max_retry_attempts:
            stopping_reason = "max_retries_reached"
        
        return {
            "final_prompt": current_prompt,
            "final_metrics": current_metrics,
            "optimization_history": optimization_history,
            "total_iterations": iteration - 1,
            "improvement_history": improvement_history,
            "stopping_reason": stopping_reason,
            "human_feedback_summary": feedback_summary,
            "human_feedback_results": human_feedback_results
        }
    
    async def _evaluate_candidates_dev_a(
        self,
        candidates: List[Dict[str, Any]],
        dev_a_data: List[Dict[str, Any]],
        baseline_metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate all candidates on Dev A and return the best one using comprehensive scoring
        """
        best_candidate = None
        best_score = -float('inf')
        baseline_accuracy = baseline_metrics['overall_accuracy']
        baseline_f1 = baseline_metrics['summary']['average_enum_macro_f1']
        baseline_valid_json = baseline_metrics['validation_metrics']['valid_json_accuracy']
        baseline_failed_cases = len(baseline_metrics['detailed_failed_cases']['wrong_classifications'])
        
        for candidate in candidates:
            print(f"   Evaluating: {candidate['strategy']}")
            
            # Evaluate candidate
            candidate_metrics = await self._get_evaluation_engine().evaluate_prompt(
                prompt=candidate['optimized_prompt'],
                data=dev_a_data,
                schema=self.schema,
                evaluation_type="dev_a_candidate"
            )
            
            # Calculate comprehensive score
            accuracy_improvement = candidate_metrics['overall_accuracy'] - baseline_accuracy
            f1_improvement = candidate_metrics['summary']['average_enum_macro_f1'] - baseline_f1
            valid_json_improvement = candidate_metrics['validation_metrics']['valid_json_accuracy'] - baseline_valid_json
            failed_cases_reduction = baseline_failed_cases - len(candidate_metrics['detailed_failed_cases']['wrong_classifications'])
            
            # Weighted composite score (prioritizing different aspects)
            composite_score = (
                accuracy_improvement * 3.0 +      # Primary metric (weight: 3)
                f1_improvement * 2.0 +            # F1 score (weight: 2)
                valid_json_improvement * 1.5 +    # JSON validity (weight: 1.5)
                (failed_cases_reduction / max(baseline_failed_cases, 1)) * 1.0  # Failed cases reduction (weight: 1)
            )
            
            print(f"     Accuracy: {candidate_metrics['overall_accuracy']:.3f} (Δ{accuracy_improvement:+.3f})")
            print(f"     F1 Score: {candidate_metrics['summary']['average_enum_macro_f1']:.3f} (Δ{f1_improvement:+.3f})")
            print(f"     Valid JSON: {candidate_metrics['validation_metrics']['valid_json_accuracy']:.3f} (Δ{valid_json_improvement:+.3f})")
            print(f"     Failed Cases: {len(candidate_metrics['detailed_failed_cases']['wrong_classifications'])} (Δ{-failed_cases_reduction:+d})")
            print(f"     Composite Score: {composite_score:+.3f}")
            
            # Select candidate with best composite score (even if accuracy doesn't improve)
            if composite_score > best_score:
                best_score = composite_score
                best_candidate = {
                    **candidate,
                    'dev_a_metrics': candidate_metrics,
                    'improvement': accuracy_improvement,  # Keep for backward compatibility
                    'composite_score': composite_score,
                    'f1_improvement': f1_improvement,
                    'valid_json_improvement': valid_json_improvement,
                    'failed_cases_reduction': failed_cases_reduction
                }
        
        return best_candidate
    
    async def _run_final_evaluation(
        self,
        optimization_results: Dict[str, Any],
        train_baseline_metrics: Dict[str, Any],
        dev_a_baseline_metrics: Dict[str, Any],
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run final evaluation with Dev A comparison and test data for hidden validation
        """
        final_prompt = optimization_results['final_prompt']
        
        # Get the final Dev A metrics from optimization results (this is what final_metrics represents)
        dev_a_optimized_metrics = optimization_results['final_metrics']
        
        # Evaluate final prompt on test data for hidden validation
        print("📊 Evaluating final prompt on test data (hidden validation)...")
        test_metrics = await self._get_evaluation_engine().evaluate_prompt(
            prompt=final_prompt,
            data=test_data,
            schema=self.schema,
            evaluation_type="final_test"
        )
        
        # Evaluate baseline on test data for comparison
        print("📊 Evaluating baseline prompt on test data...")
        baseline_test_metrics = await self._get_evaluation_engine().evaluate_prompt(
            prompt=self._get_baseline_prompt(),
            data=test_data,
            schema=self.schema,
            evaluation_type="baseline_test"
        )
        
        # Calculate Dev A comparison metrics (primary optimization target)
        dev_a_accuracy_improvement = dev_a_optimized_metrics['overall_accuracy'] - dev_a_baseline_metrics['overall_accuracy']
        dev_a_f1_improvement = dev_a_optimized_metrics['summary']['average_enum_macro_f1'] - dev_a_baseline_metrics['summary']['average_enum_macro_f1']
        dev_a_valid_json_improvement = dev_a_optimized_metrics['validation_metrics']['valid_json_accuracy'] - dev_a_baseline_metrics['validation_metrics']['valid_json_accuracy']
        
        dev_a_baseline_failed_cases = len(dev_a_baseline_metrics['detailed_failed_cases']['wrong_classifications'])
        dev_a_optimized_failed_cases = len(dev_a_optimized_metrics['detailed_failed_cases']['wrong_classifications'])
        dev_a_failed_cases_reduction = dev_a_baseline_failed_cases - dev_a_optimized_failed_cases
        
        # Calculate comprehensive score for Dev A (primary metric for deployment decision)
        dev_a_comprehensive_score = (
            dev_a_accuracy_improvement * 3.0 +      # Primary metric (weight: 3)
            dev_a_f1_improvement * 2.0 +            # F1 score (weight: 2)
            dev_a_valid_json_improvement * 1.5 +    # JSON validity (weight: 1.5)
            (dev_a_failed_cases_reduction / max(dev_a_baseline_failed_cases, 1)) * 1.0  # Failed cases reduction (weight: 1)
        )
        
        # Calculate test data metrics for hidden validation
        test_accuracy_improvement = test_metrics['overall_accuracy'] - baseline_test_metrics['overall_accuracy']
        test_f1_improvement = test_metrics['summary']['average_enum_macro_f1'] - baseline_test_metrics['summary']['average_enum_macro_f1']
        test_valid_json_improvement = test_metrics['validation_metrics']['valid_json_accuracy'] - baseline_test_metrics['validation_metrics']['valid_json_accuracy']
        
        test_baseline_failed_cases = len(baseline_test_metrics['detailed_failed_cases']['wrong_classifications'])
        test_optimized_failed_cases = len(test_metrics['detailed_failed_cases']['wrong_classifications'])
        test_failed_cases_reduction = test_baseline_failed_cases - test_optimized_failed_cases
        
        print(f"\n📈 Dev A Optimization Results (Primary Target):")
        print(f"   Baseline Dev A Accuracy: {dev_a_baseline_metrics['overall_accuracy']:.3f}")
        print(f"   Optimized Dev A Accuracy: {dev_a_optimized_metrics['overall_accuracy']:.3f}")
        print(f"   Dev A Accuracy Improvement: {dev_a_accuracy_improvement:+.3f}")
        print(f"   Dev A F1 Score Improvement: {dev_a_f1_improvement:+.3f}")
        print(f"   Dev A Valid JSON Improvement: {dev_a_valid_json_improvement:+.3f}")
        print(f"   Dev A Failed Cases Reduction: {dev_a_failed_cases_reduction:+d}")
        print(f"   Dev A Comprehensive Score: {dev_a_comprehensive_score:+.3f}")
        
        print(f"\n📊 Test Data Results (Hidden Validation):")
        print(f"   Baseline Test Accuracy: {baseline_test_metrics['overall_accuracy']:.3f}")
        print(f"   Optimized Test Accuracy: {test_metrics['overall_accuracy']:.3f}")
        print(f"   Test Accuracy Improvement: {test_accuracy_improvement:+.3f}")
        print(f"   Test F1 Score Improvement: {test_f1_improvement:+.3f}")
        print(f"   Test Valid JSON Improvement: {test_valid_json_improvement:+.3f}")
        print(f"   Test Failed Cases Reduction: {test_failed_cases_reduction:+d}")
        
        # Enhanced deployment decision based on Dev A comprehensive score (primary target)
        if dev_a_comprehensive_score > 0.01:  # Positive comprehensive improvement on Dev A
            deployment_decision = "DEPLOY"
            recommended_prompt = final_prompt
            print("✅ RECOMMENDATION: Deploy optimized prompt (Dev A comprehensive improvement)")
        elif dev_a_accuracy_improvement > 0.005:  # Small but positive accuracy improvement on Dev A
            deployment_decision = "DEPLOY"
            recommended_prompt = final_prompt
            print("✅ RECOMMENDATION: Deploy optimized prompt (Dev A accuracy improvement)")
        elif dev_a_f1_improvement > 0.02 and dev_a_valid_json_improvement >= 0:  # Good F1 improvement with no JSON regression on Dev A
            deployment_decision = "DEPLOY"
            recommended_prompt = final_prompt
            print("✅ RECOMMENDATION: Deploy optimized prompt (Dev A F1 improvement)")
        elif dev_a_failed_cases_reduction > 0 and dev_a_accuracy_improvement >= -0.01:  # Fewer failures with minimal accuracy loss on Dev A
            deployment_decision = "DEPLOY"
            recommended_prompt = final_prompt
            print("✅ RECOMMENDATION: Deploy optimized prompt (Dev A fewer failed cases)")
        else:
            deployment_decision = "KEEP_BASELINE"
            recommended_prompt = self._get_baseline_prompt()
            print("⚠️  RECOMMENDATION: Keep baseline prompt (no significant Dev A improvement)")
        
        # Save final recommended prompt
        await self._save_final_prompt(recommended_prompt, deployment_decision, dev_a_comprehensive_score)
        
        return {
            "deployment_decision": deployment_decision,
            "recommended_prompt": recommended_prompt,
            # Dev A metrics (primary optimization target)
            "dev_a_improvement": dev_a_accuracy_improvement,  # Primary comparison metric
            "dev_a_comprehensive_score": dev_a_comprehensive_score,
            "dev_a_accuracy_improvement": dev_a_accuracy_improvement,
            "dev_a_f1_improvement": dev_a_f1_improvement,
            "dev_a_valid_json_improvement": dev_a_valid_json_improvement,
            "dev_a_failed_cases_reduction": dev_a_failed_cases_reduction,
            "dev_a_baseline_metrics": dev_a_baseline_metrics,
            "dev_a_optimized_metrics": dev_a_optimized_metrics,
            # Test metrics (hidden validation)
            "test_improvement": test_accuracy_improvement,  # Keep for backward compatibility
            "test_comprehensive_score": (test_accuracy_improvement * 3.0 + test_f1_improvement * 2.0 + test_valid_json_improvement * 1.5 + (test_failed_cases_reduction / max(test_baseline_failed_cases, 1)) * 1.0),
            "test_accuracy_improvement": test_accuracy_improvement,
            "test_f1_improvement": test_f1_improvement,
            "test_valid_json_improvement": test_valid_json_improvement,
            "test_failed_cases_reduction": test_failed_cases_reduction,
            "baseline_test_metrics": baseline_test_metrics,
            "optimized_test_metrics": test_metrics,
            # Original data for reference
            "train_baseline_metrics": train_baseline_metrics,
            "optimization_results": optimization_results,
            # Backward compatibility (use Dev A as primary)
            "comprehensive_score": dev_a_comprehensive_score,
            "accuracy_improvement": dev_a_accuracy_improvement,
            "f1_improvement": dev_a_f1_improvement,
            "valid_json_improvement": dev_a_valid_json_improvement,
            "failed_cases_reduction": dev_a_failed_cases_reduction
        }
    
    async def _save_baseline_prompt(self, request_id: str):
        """Save the baseline prompt to intermediate results folder"""
        try:
            baseline_dir = f"fastapi_optimization_system/intermediate_results/{request_id}/baseline"
            os.makedirs(baseline_dir, exist_ok=True)
            
            baseline_prompt = self._get_baseline_prompt()
            
            # Save baseline prompt metadata
            baseline_data = {
                "timestamp": datetime.now().isoformat(),
                "prompt_type": "baseline",
                "schema": self.schema,
                "baseline_prompt": baseline_prompt
            }
            
            # Save as JSON
            baseline_file = f"{baseline_dir}/baseline_prompt.json"
            with open(baseline_file, "w") as f:
                json.dump(baseline_data, f, indent=2, default=str)
            
            # Save as readable text file
            prompt_file = f"{baseline_dir}/baseline_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write("BASELINE PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Schema: {json.dumps(self.schema, indent=2)}\n")
                f.write("\nBASELINE PROMPT:\n")
                f.write("=" * 80 + "\n")
                f.write(baseline_prompt)
            
            print(f"💾 Saved baseline prompt to: {baseline_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save baseline prompt: {e}")
    
    async def _save_candidate_prompts(self, candidates: List[Dict[str, Any]], iteration: int):
        """Save all generated candidate prompts to intermediate results folder"""
        try:
            request_id = get_request_id()
            candidates_dir = f"fastapi_optimization_system/intermediate_results/{request_id}/iteration_{iteration:02d}_candidates"
            os.makedirs(candidates_dir, exist_ok=True)
            
            # Save all candidates in a single file
            candidates_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "total_candidates": len(candidates),
                "candidates": candidates
            }
            
            candidates_file = f"{candidates_dir}/all_candidates.json"
            with open(candidates_file, "w") as f:
                json.dump(candidates_data, f, indent=2, default=str)
            
            # Save individual prompt files for easy reading
            for i, candidate in enumerate(candidates):
                strategy = candidate.get('strategy', f'candidate_{i}')
                prompt_file = f"{candidates_dir}/{i+1:02d}_{strategy}_prompt.txt"
                
                with open(prompt_file, "w") as f:
                    f.write(f"Strategy: {candidate.get('strategy', 'Unknown')}\n")
                    f.write(f"Confidence: {candidate.get('confidence', 0.0):.3f}\n")
                    f.write(f"Execution Time: {candidate.get('execution_time', 0.0):.2f}s\n")
                    f.write(f"Changes Made: {', '.join(candidate.get('changes_made', []))}\n")
                    f.write(f"Reasoning: {candidate.get('reasoning', 'No reasoning provided')}\n")
                    f.write("-" * 80 + "\n")
                    f.write("OPTIMIZED PROMPT:\n")
                    f.write("-" * 80 + "\n")
                    f.write(candidate.get('optimized_prompt', ''))
            
            print(f"💾 Saved {len(candidates)} candidate prompts to: {candidates_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save candidate prompts: {e}")
    
    async def _save_selected_prompt(self, best_candidate: Dict[str, Any], iteration: int):
        """Save the selected best prompt to intermediate results folder"""
        try:
            request_id = get_request_id()
            selected_dir = f"fastapi_optimization_system/intermediate_results/{request_id}/iteration_{iteration:02d}_selected"
            os.makedirs(selected_dir, exist_ok=True)
            
            # Save selected prompt metadata
            selected_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "strategy": best_candidate.get('strategy', 'Unknown'),
                "confidence": best_candidate.get('confidence', 0.0),
                "dev_a_improvement": best_candidate.get('improvement', 0.0),
                "execution_time": best_candidate.get('execution_time', 0.0),
                "changes_made": best_candidate.get('changes_made', []),
                "reasoning": best_candidate.get('reasoning', ''),
                "dev_a_metrics": best_candidate.get('dev_a_metrics', {}),
                "optimized_prompt": best_candidate.get('optimized_prompt', '')
            }
            
            # Save as JSON
            selected_file = f"{selected_dir}/selected_prompt.json"
            with open(selected_file, "w") as f:
                json.dump(selected_data, f, indent=2, default=str)
            
            # Save as readable text file
            prompt_file = f"{selected_dir}/selected_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(f"ITERATION {iteration} - SELECTED PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Strategy: {best_candidate.get('strategy', 'Unknown')}\n")
                f.write(f"Confidence: {best_candidate.get('confidence', 0.0):.3f}\n")
                f.write(f"Dev A Improvement: {best_candidate.get('improvement', 0.0):+.3f}\n")
                f.write(f"Execution Time: {best_candidate.get('execution_time', 0.0):.2f}s\n")
                f.write(f"Changes Made: {', '.join(best_candidate.get('changes_made', []))}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("\nReasoning:\n")
                f.write("-" * 40 + "\n")
                f.write(best_candidate.get('reasoning', 'No reasoning provided'))
                f.write("\n\n")
                f.write("OPTIMIZED PROMPT:\n")
                f.write("=" * 80 + "\n")
                f.write(best_candidate.get('optimized_prompt', ''))
            
            print(f"💾 Saved selected prompt to: {selected_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save selected prompt: {e}")
    
    async def _save_final_prompt(self, recommended_prompt: str, deployment_decision: str, test_improvement: float):
        """Save the final recommended prompt to intermediate results folder"""
        try:
            request_id = get_request_id()
            final_dir = f"fastapi_optimization_system/intermediate_results/{request_id}/final"
            os.makedirs(final_dir, exist_ok=True)
            
            # Save final prompt metadata
            final_data = {
                "timestamp": datetime.now().isoformat(),
                "deployment_decision": deployment_decision,
                "test_improvement": test_improvement,
                "recommended_prompt": recommended_prompt
            }
            
            # Save as JSON
            final_file = f"{final_dir}/final_prompt.json"
            with open(final_file, "w") as f:
                json.dump(final_data, f, indent=2, default=str)
            
            # Save as readable text file
            prompt_file = f"{final_dir}/final_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write("FINAL RECOMMENDED PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Deployment Decision: {deployment_decision}\n")
                f.write(f"Test Improvement: {test_improvement:+.3f}\n")
                f.write("\nRECOMMENDED PROMPT:\n")
                f.write("=" * 80 + "\n")
                f.write(recommended_prompt)
            
            print(f"💾 Saved final recommended prompt to: {final_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save final prompt: {e}")
    
    def _get_baseline_prompt(self) -> str:
        """Get the baseline prompt"""
        # add schema and this important note, first line of the prompt
        return self.baseline_prompt + "\n\n" + "The json schema with the fields and their possible values (enum values) is as follows:\n" + json.dumps(self.schema, indent=2) + "\n\nIMPORTANT: Respond with a valid JSON object only. The value should be from enum values only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON."


async def main():
    """Main entry point"""
    system = CompleteOptimizationSystem()
    
    try:
        results = await system.run_complete_optimization()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        req_id = get_request_id()
        results_file = f"fastapi_optimization_system/results/optimization_results_{timestamp}_{req_id}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print final summary
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Decision: {results['deployment_decision']}")
        print(f"   Accuracy Improvement: {results['test_improvement']:+.3f}")
        if 'comprehensive_score' in results:
            print(f"   Comprehensive Score: {results['comprehensive_score']:+.3f}")
        if 'f1_improvement' in results:
            print(f"   F1 Score Improvement: {results['f1_improvement']:+.3f}")
        if 'valid_json_improvement' in results:
            print(f"   Valid JSON Improvement: {results['valid_json_improvement']:+.3f}")
        if 'failed_cases_reduction' in results:
            print(f"   Failed Cases Reduction: {results['failed_cases_reduction']:+d}")
        print(f"   Total Iterations: {results['optimization_results']['total_iterations']}")
        print(f"   Stopping Reason: {results['optimization_results']['stopping_reason']}")
        
        # Print improvement history if available
        if 'improvement_history' in results['optimization_results']:
            print(f"   Improvement History: {[f'{imp:.3f}' for imp in results['optimization_results']['improvement_history']]}")
            
        # Explain stopping reason
        stopping_reason = results['optimization_results']['stopping_reason']
        if stopping_reason == "converged":
            print("   ✅ Optimization converged - improvements stabilized")
        elif stopping_reason == "max_retries_reached":
            print("   ⚠️  Optimization stopped - maximum retry attempts reached")
        elif stopping_reason == "below_threshold":
            print("   ⏹️  Optimization stopped - improvement below threshold")
        elif stopping_reason == "max_iterations_reached":
            print("   🔄 Optimization stopped - maximum iterations reached")
        
    except Exception as e:
        print(f"❌ System failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 