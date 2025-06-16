import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import uuid
import asyncio
import json
from pathlib import Path

# Add the parent directory to sys.path to import existing components
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from prompt_optimizer.main import CompleteOptimizationSystem
from prompt_optimizer.core.data_manager import DataManager
from prompt_optimizer.core.evaluation_engine import EvaluationEngine
from prompt_optimizer.core.optimization_controller import OptimizationController
from prompt_optimizer.core.enhanced_human_feedback_fixed import create_simple_human_feedback_manager

from app.models.optimization_models import (
    OptimizationRequest, 
    OptimizationResult, 
    OptimizationProgress,
    DataSplitSummary,
    OptimizationIterationResult,
    HumanFeedbackSummary
)
from app.utils.error_handler import (
    OptimizationError, 
    DataProcessingError, 
    OptimizationProcessError,
    ModelConfigurationError,
    DatasetError,
    handle_optimization_exception
)
from app.utils.context_util import get_request_id
from app.config.settings import get_settings

class OptimizationService:
    """
    Service layer that wraps the existing complete_optimization_system
    Provides FastAPI integration without changing core business logic
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._active_optimizations: Dict[str, Dict[str, Any]] = {}
        
        # Initialize components (reusing existing system)
        self.complete_system = CompleteOptimizationSystem()
        
    async def start_optimization(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Start the complete optimization process
        This is the main entry point that handles the entire flow
        """
        request_id = get_request_id() or str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Store optimization request
            self._active_optimizations[request_id] = {
                "request": request,
                "status": "started",
                "start_time": start_time,
                "current_step": "initialization"
            }
            
            # Convert FastAPI request to internal format
            optimization_config = self._convert_request_to_config(request)
            
            # Update progress
            await self._update_progress(request_id, "data_preparation", 10.0, "Preparing data splits...")
            
            # Run the complete optimization using existing system
            results = await self._run_complete_optimization(request_id, optimization_config)
            
            # Convert results to FastAPI response format
            optimization_result = self._convert_results_to_response(
                request_id, request, results, start_time
            )
            
            # Update status
            self._active_optimizations[request_id]["status"] = "completed"
            
            return optimization_result
            
        except Exception as e:
            # Handle any errors
            self._active_optimizations[request_id]["status"] = "failed"
            optimization_error = handle_optimization_exception(e, request_id, "optimization")
            raise optimization_error
    
    async def get_optimization_progress(self, request_id: str) -> OptimizationProgress:
        """Get current optimization progress"""
        if request_id not in self._active_optimizations:
            raise OptimizationError(
                f"Optimization request {request_id} not found",
                status_code=404,
                request_id=request_id
            )
        
        opt_data = self._active_optimizations[request_id]
        
        return OptimizationProgress(
            request_id=request_id,
            current_step=opt_data.get("current_step", "unknown"),
            progress_percentage=opt_data.get("progress_percentage", 0.0),
            estimated_time_remaining=opt_data.get("estimated_time_remaining"),
            current_iteration=opt_data.get("current_iteration"),
            total_iterations=opt_data.get("total_iterations"),
            message=opt_data.get("message", "Processing...")
        )
    
    def _convert_request_to_config(self, request: OptimizationRequest) -> Dict[str, Any]:
        """Convert FastAPI request to internal optimization configuration"""
        return {
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "schema": request.json_schema,
            "model_config": {
                "provider": request.model_configuration.provider.value,
                "model_name": request.model_configuration.model_name,
                "temperature": request.model_configuration.temperature,
                "max_tokens": request.model_configuration.max_tokens
            },
            "dataset": request.dataset,
            "max_iterations": request.max_iterations,
            "improvement_threshold": request.improvement_threshold,
            "enable_human_feedback": request.enable_human_feedback
        }
    
    async def _run_complete_optimization(
        self, 
        request_id: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the complete optimization using existing system
        This method integrates with your existing complete_optimization_system
        """
        try:
            # Initialize request ID in the existing system
            from prompt_optimizer.core.request_id import initialize_request_id
            initialize_request_id()
            
            # Create intermediate results directory and copy baseline files
            await self._create_intermediate_results_dir(request_id)
            await self._copy_baseline_files_to_request_folder(request_id)
            
            # Set up the enhanced system prompt (only system prompt gets optimized)
            enhanced_system_prompt = self._create_enhanced_system_prompt(
                config['system_prompt'], 
                config['schema']
            )
            
            # User prompt stays separate as the query template
            user_prompt_template = config['user_prompt']
            
            # Save baseline prompt
            await self._save_baseline_prompt(request_id, enhanced_system_prompt, user_prompt_template, config['schema'])
            
            # Update progress
            await self._update_progress(request_id, "data_loading", 20.0, "Loading and splitting dataset...")
            
            # Use existing data manager
            data_manager = DataManager()
            data_splits = await data_manager.prepare_data_splits(config['dataset'])
            
            # Save data splits
            await self._save_data_splits(request_id, data_splits)
            
            # Update progress
            await self._update_progress(request_id, "baseline_evaluation", 30.0, "Evaluating baseline performance...")
            
            # Use existing evaluation engine
            evaluation_engine = EvaluationEngine()
            
            # Evaluate baseline on train data (system + user prompt separately)
            train_baseline_metrics = await evaluation_engine.evaluate_prompt(
                enhanced_system_prompt,
                data_splits['train'],
                config['schema'],
                "train_baseline",
                user_prompt_template=user_prompt_template
            )
            
            # Save train baseline metrics
            await self._save_train_baseline_metrics(request_id, train_baseline_metrics)
            
            # Save in the format expected by existing components (for compatibility)
            await self._save_enhanced_baseline_results(request_id, train_baseline_metrics)
            
            # Evaluate baseline on dev_a (hidden comparison dataset)
            dev_a_baseline_metrics = await evaluation_engine.evaluate_prompt(
                enhanced_system_prompt,
                data_splits['dev_a'],
                config['schema'],
                "dev_a_baseline",
                user_prompt_template=user_prompt_template
            )
            
            # Save dev_a baseline metrics
            await self._save_dev_a_baseline_metrics(request_id, dev_a_baseline_metrics)
            
            # Update progress
            await self._update_progress(request_id, "intent_analysis", 40.0, "Analyzing optimization intent...")
            
            # Use existing optimization controller for intent analysis
            optimization_controller = OptimizationController()
            intent_analysis = await optimization_controller.analyze_intent(
                config['schema'],
                train_baseline_metrics,
                data_splits['train'][:5],  # Sample for intent analysis
                enhanced_system_prompt,
                user_prompt_template=user_prompt_template
            )
            
            # Save intent analysis
            await self._save_intent_analysis(request_id, intent_analysis)
            
            # Update progress
            await self._update_progress(request_id, "optimization_loop", 50.0, "Starting optimization iterations...")
            
            # Run the optimization loop (same as complete system)
            optimization_results = await self._run_optimization_loop(
                request_id,
                enhanced_system_prompt,
                user_prompt_template,
                train_baseline_metrics,
                dev_a_baseline_metrics,
                intent_analysis,
                data_splits,
                config
            )
            
            # Save optimization results
            await self._save_optimization_results(request_id, optimization_results)
            
            # Update progress
            await self._update_progress(request_id, "final_evaluation", 90.0, "Running final evaluation...")
            
            # Final evaluation on test data
            final_system_prompt = optimization_results.get('recommended_prompt', enhanced_system_prompt)
            test_metrics = await evaluation_engine.evaluate_prompt(
                final_system_prompt,
                data_splits['test'],
                config['schema'],
                "test_final",
                user_prompt_template=user_prompt_template
            )
            
            # Save test metrics
            await self._save_test_metrics(request_id, test_metrics)
            
            # Update progress
            await self._update_progress(request_id, "completed", 100.0, "Optimization completed!")
            
            return {
                "data_splits": data_splits,
                "baseline_metrics": train_baseline_metrics,
                "dev_a_baseline_metrics": dev_a_baseline_metrics,
                "intent_analysis": intent_analysis,
                "optimization_results": optimization_results,
                "test_metrics": test_metrics,
                "baseline_system_prompt": enhanced_system_prompt,
                "user_prompt_template": user_prompt_template
            }
            
        except Exception as e:
            raise OptimizationProcessError(
                f"Optimization process failed: {str(e)}",
                request_id=request_id
            )
    
    async def _run_optimization_loop(
        self,
        request_id: str,
        baseline_system_prompt: str,
        user_prompt_template: str,
        train_baseline_metrics: Dict[str, Any],
        dev_a_baseline_metrics: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        data_splits: Dict[str, List],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the optimization iteration loop (same as complete system)"""
        
        optimization_controller = OptimizationController()
        evaluation_engine = EvaluationEngine()
        enhanced_human_feedback = create_simple_human_feedback_manager()
        
        current_system_prompt = baseline_system_prompt
        current_metrics = train_baseline_metrics
        optimization_history = []
        best_candidate = None
        
        max_iterations = config.get('max_iterations', 5)
        improvement_threshold = config.get('improvement_threshold', 0.05)
        
        # Tracking for improved stopping mechanism (same as complete system)
        improvement_history = []
        convergence_threshold = 0.005  # Within 0.5% improvement considered same
        convergence_patience = 3  # Stop if stable for 3 iterations
        retry_attempts = 0
        max_retry_attempts = 2  # Maximum retries when no improvement found
        
        for iteration in range(1, max_iterations + 1):
            try:
                # Update progress
                progress = 50.0 + (iteration / max_iterations) * 35.0  # From 50% to 85%
                await self._update_progress(
                    request_id, 
                    f"optimization_iteration_{iteration}", 
                    progress,
                    f"Running optimization iteration {iteration}/{max_iterations}...",
                    current_iteration=iteration,
                    total_iterations=max_iterations
                )
                
                print(f"\n🔄 Optimization Iteration {iteration}")
                if retry_attempts > 0:
                    print(f"   (Retry attempt {retry_attempts}/{max_retry_attempts})")
                print("-" * 50)
                
                # Get human feedback from previous iteration if available
                previous_human_feedback = None
                if iteration > 1 and optimization_history:
                    previous_iteration = optimization_history[-1]  # Get last iteration
                    previous_human_feedback = previous_iteration.get('human_feedback_summary')
                
                # Generate candidate system prompts (only system prompt gets optimized)
                print("📝 Generating candidate prompts...")
                candidates = await optimization_controller.generate_candidates(
                    current_system_prompt,
                    current_metrics,
                    intent_analysis,
                    config['schema'],
                    iteration,
                    human_feedback_summary=previous_human_feedback,
                    optimization_history=optimization_history,
                    user_prompt_template=user_prompt_template
                )
                
                # Save generated candidate prompts
                if candidates:
                    await self._save_candidate_prompts(request_id, candidates, iteration)
                
                if not candidates:
                    print(f"❌ No candidates generated for iteration {iteration}")
                    if retry_attempts < max_retry_attempts:
                        retry_attempts += 1
                        print(f"🔄 Retrying optimization (attempt {retry_attempts}/{max_retry_attempts})...")
                        continue
                    else:
                        print("❌ Maximum retry attempts reached, stopping optimization")
                        break
                
                print(f"Generated {len(candidates)} candidates for iteration {iteration}")
                
                # Reset retry attempts on successful candidate generation
                retry_attempts = 0
                
                # Evaluate candidates on dev_a using comprehensive scoring (same as complete system)
                print("📊 Evaluating candidates on Dev A...")
                best_iteration_candidate = await self._evaluate_candidates_dev_a(
                    candidates=candidates,
                    dev_a_data=data_splits['dev_a'],
                    baseline_metrics=dev_a_baseline_metrics,
                    evaluation_engine=evaluation_engine,
                    config=config,
                    user_prompt_template=user_prompt_template,
                    iteration=iteration
                )
                
                if best_iteration_candidate is None:
                    print(f"❌ No suitable candidate found in iteration {iteration}")
                    if retry_attempts < max_retry_attempts:
                        retry_attempts += 1
                        print(f"🔄 Retrying optimization (attempt {retry_attempts}/{max_retry_attempts})...")
                        continue
                    else:
                        print("❌ Maximum retry attempts reached, stopping optimization")
                        break
                
                # Save the selected best prompt
                await self._save_selected_prompt(request_id, best_iteration_candidate, iteration)
                
                # Add improvement to history for convergence tracking (use composite score for better tracking)
                composite_score = best_iteration_candidate.get('composite_score', best_iteration_candidate['improvement_over_baseline'])
                improvement_history.append(composite_score)
                
                # Adaptive stopping criteria (same as complete system)
                should_continue = self._check_stopping_criteria(
                    best_iteration_candidate, iteration, improvement_history, 
                    improvement_threshold, convergence_threshold, convergence_patience
                )
                
                if not should_continue and iteration >= 2:
                    print("⏹️  Stopping optimization based on stopping criteria")
                    break
                
                # Run candidate on dev_b for human feedback
                best_candidate_prompt = (best_iteration_candidate.get('candidate_prompt') or 
                                       best_iteration_candidate.get('prompt') or 
                                       best_iteration_candidate.get('optimized_prompt'))
                
                print("🔬 Evaluating on Dev B...")
                dev_b_metrics = await evaluation_engine.evaluate_prompt(
                    best_candidate_prompt,  # Optimized system prompt
                    data_splits['dev_b'],
                    config['schema'],
                    f"dev_b_iteration_{iteration}",
                    user_prompt_template=user_prompt_template
                )
                
                print(f"✅ Dev B evaluation completed:")
                print(f"   Overall Accuracy: {dev_b_metrics['overall_accuracy']:.3f}")
                print(f"   Average F1: {dev_b_metrics['summary']['average_enum_macro_f1']:.3f}")
                
                # Collect enhanced human feedback (same as complete system)
                human_feedback_summary = None
                human_feedback_details = None
                if config.get('enable_human_feedback', True):
                    try:
                        print("👥 Collecting enhanced human feedback...")
                        feedback_summary, human_feedback_results = await enhanced_human_feedback.collect_human_feedback_complete_workflow(
                            candidate_prompt=best_candidate_prompt,
                            dev_b_results=dev_b_metrics,
                            iteration=iteration,
                            baseline_metrics=dev_a_baseline_metrics  # Use dev A baseline for comparison
                        )
                        human_feedback_summary = feedback_summary
                        human_feedback_details = human_feedback_results
                        
                        # Print human feedback summary for transparency (same as complete system)
                        if feedback_summary.total_cases > 0:
                            print(f"📊 Human Feedback Summary:")
                            print(f"   Correct: {feedback_summary.correct_count}, Incorrect: {feedback_summary.incorrect_count}, Skipped: {feedback_summary.skipped_count}")
                            print(f"   Reviewer confidence: {feedback_summary.average_confidence:.2f}")
                            if feedback_summary.key_feedback_themes:
                                print(f"   Key themes: {', '.join(feedback_summary.key_feedback_themes[:3])}")
                            if feedback_summary.improvement_suggestions:
                                print(f"   Suggestions: {feedback_summary.improvement_suggestions[0][:50]}...")
                        
                    except Exception as e:
                        print(f"Human feedback collection failed: {e}")
                
                # Update for next iteration (only system prompt changes)
                current_system_prompt = best_candidate_prompt
                current_metrics = best_iteration_candidate['dev_a_metrics']
                
                # Store iteration results with enhanced feedback (same as complete system)
                iteration_result = {
                    "iteration": iteration,
                    "strategy": best_iteration_candidate.get('strategy', 'unknown'),
                    "dev_a_improvement": best_iteration_candidate['improvement_over_baseline'],
                    "dev_b_metrics": dev_b_metrics,
                    "human_feedback_summary": human_feedback_summary,  # Summarized feedback for context
                    "human_feedback_details": human_feedback_details,  # Detailed results for analysis
                    "optimized_prompt": best_candidate_prompt,
                    "candidate": best_iteration_candidate
                }
                optimization_history.append(iteration_result)
                
                # Update current best using composite score (same as complete system)
                current_best_score = best_candidate.get('composite_score', -float('inf')) if best_candidate else -float('inf')
                if best_iteration_candidate['composite_score'] > current_best_score:
                    best_candidate = best_iteration_candidate.copy()
                    best_candidate['iteration'] = iteration
                    best_candidate['dev_b_metrics'] = dev_b_metrics
                    best_candidate['human_feedback_summary'] = human_feedback_summary
                    best_candidate['human_feedback_details'] = human_feedback_details
                    
                    # Log the selection (same format as complete system)
                    print(f"✅ Best candidate selected: {best_candidate.get('strategy', 'unknown')}")
                    print(f"   Dev A Accuracy Improvement: {best_candidate['improvement_over_baseline']:.3f}")
                    print(f"   Composite Score: {best_candidate['composite_score']:+.3f}")
                    if best_candidate.get('f1_improvement', 0) != 0:
                        print(f"   F1 Score Improvement: {best_candidate['f1_improvement']:+.3f}")
                    if best_candidate.get('valid_json_improvement', 0) != 0:
                        print(f"   Valid JSON Improvement: {best_candidate['valid_json_improvement']:+.3f}")
                    if best_candidate.get('failed_cases_reduction', 0) != 0:
                        print(f"   Failed Cases Reduction: {best_candidate['failed_cases_reduction']:+d}")
                
                print(f"✅ Iteration {iteration} completed")
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                break
        
        # Add stopping reason to results (same as complete system)
        stopping_reason = "max_iterations_reached"
        if len(improvement_history) >= convergence_patience:
            recent_improvements = improvement_history[-convergence_patience:]
            min_improvement = min(recent_improvements)
            max_improvement = max(recent_improvements)
            if max_improvement - min_improvement <= convergence_threshold:
                stopping_reason = "converged"
        elif improvement_history and improvement_history[-1] < improvement_threshold:
            stopping_reason = "below_threshold"
        elif retry_attempts >= max_retry_attempts:
            stopping_reason = "max_retries_reached"
        
        # Get the final recommended prompt
        final_recommended_prompt = baseline_system_prompt
        if best_candidate:
            final_recommended_prompt = (best_candidate.get('candidate_prompt') or 
                                      best_candidate.get('prompt') or 
                                      best_candidate.get('optimized_prompt') or 
                                      baseline_system_prompt)
        
        return {
            "final_prompt": current_system_prompt,
            "final_metrics": current_metrics,
            "optimization_history": optimization_history,
            "total_iterations": len(optimization_history),
            "improvement_history": improvement_history,
            "stopping_reason": stopping_reason,
            "human_feedback_summary": human_feedback_summary if 'human_feedback_summary' in locals() else None,
            "human_feedback_results": human_feedback_details if 'human_feedback_details' in locals() else None,
            'best_candidate': best_candidate,
            'recommended_prompt': final_recommended_prompt,
            'final_improvement': best_candidate.get('improvement_over_baseline', 0.0) if best_candidate else 0.0
        }
    
    async def _evaluate_candidates_dev_a(
        self,
        candidates: List[Dict[str, Any]],
        dev_a_data: List[Dict[str, Any]],
        baseline_metrics: Dict[str, Any],
        evaluation_engine,
        config: Dict[str, Any],
        user_prompt_template: str,
        iteration: int
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate all candidates on Dev A and return the best one using comprehensive scoring
        (Same logic as complete system)
        """
        best_candidate = None
        best_score = -float('inf')
        baseline_accuracy = baseline_metrics['overall_accuracy']
        baseline_f1 = baseline_metrics['summary']['average_enum_macro_f1']
        baseline_valid_json = baseline_metrics['validation_metrics']['valid_json_accuracy']
        baseline_failed_cases = len(baseline_metrics['detailed_failed_cases']['wrong_classifications'])
        
        for candidate in candidates:
            # Get the candidate prompt - handle different possible keys
            candidate_prompt = candidate.get('candidate_prompt') or candidate.get('prompt') or candidate.get('optimized_prompt')
            if not candidate_prompt:
                print(f"Warning: No candidate prompt found in candidate: {candidate.keys()}")
                continue
                
            print(f"   Evaluating: {candidate.get('strategy', 'unknown')}")
            
            # Evaluate candidate
            candidate_metrics = await evaluation_engine.evaluate_prompt(
                candidate_prompt,  # This is the optimized system prompt
                dev_a_data,
                config['schema'],
                f"dev_a_iteration_{iteration}",
                user_prompt_template=user_prompt_template
            )
            
            # Calculate comprehensive score (same as complete system)
            accuracy_improvement = candidate_metrics['overall_accuracy'] - baseline_accuracy
            f1_improvement = candidate_metrics['summary']['average_enum_macro_f1'] - baseline_f1
            valid_json_improvement = candidate_metrics['validation_metrics']['valid_json_accuracy'] - baseline_valid_json
            failed_cases_reduction = baseline_failed_cases - len(candidate_metrics['detailed_failed_cases']['wrong_classifications'])
            
            # Weighted composite score (prioritizing different aspects) - same weights as complete system
            composite_score = (
                accuracy_improvement * 3.0 +      # Primary metric (weight: 3)
                f1_improvement * 2.0 +            # F1 score (weight: 2)
                valid_json_improvement * 1.5 +    # JSON validity (weight: 1.5)
                (failed_cases_reduction / max(baseline_failed_cases, 1)) * 1.0  # Failed cases reduction (weight: 1)
            )
            
            # Store comprehensive metrics
            candidate['dev_a_metrics'] = candidate_metrics
            candidate['improvement_over_baseline'] = accuracy_improvement  # Keep for backward compatibility
            candidate['composite_score'] = composite_score
            candidate['f1_improvement'] = f1_improvement
            candidate['valid_json_improvement'] = valid_json_improvement
            candidate['failed_cases_reduction'] = failed_cases_reduction
            
            # Debug logging (same format as complete system)
            print(f"     Accuracy: {candidate_metrics['overall_accuracy']:.3f} (Δ{accuracy_improvement:+.3f})")
            print(f"     F1 Score: {candidate_metrics['summary']['average_enum_macro_f1']:.3f} (Δ{f1_improvement:+.3f})")
            print(f"     Valid JSON: {candidate_metrics['validation_metrics']['valid_json_accuracy']:.3f} (Δ{valid_json_improvement:+.3f})")
            print(f"     Failed Cases: {len(candidate_metrics['detailed_failed_cases']['wrong_classifications'])} (Δ{-failed_cases_reduction:+d})")
            print(f"     Composite Score: {composite_score:+.3f}")
            
            # Select candidate with best composite score (even if accuracy doesn't improve)
            if composite_score > best_score:
                best_score = composite_score
                best_candidate = candidate
        
        return best_candidate
    
    def _check_stopping_criteria(
        self,
        best_iteration_candidate: Dict[str, Any],
        iteration: int,
        improvement_history: List[float],
        improvement_threshold: float,
        convergence_threshold: float,
        convergence_patience: int
    ) -> bool:
        """
        Check stopping criteria (same logic as complete system)
        """
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
                return False
        
        # Adaptive stopping criteria: consider multiple factors (same as complete system)
        accuracy_improvement = best_iteration_candidate['improvement_over_baseline']
        composite_score = best_iteration_candidate.get('composite_score', accuracy_improvement)
        f1_improvement = best_iteration_candidate.get('f1_improvement', 0)
        failed_cases_reduction = best_iteration_candidate.get('failed_cases_reduction', 0)
        
        # Dynamic thresholds based on iteration and performance (same as complete system)
        base_threshold = improvement_threshold
        
        # Make thresholds more lenient as iterations progress (exploration vs exploitation)
        iteration_factor = max(0.3, 1.0 - (iteration * 0.1))  # Gets more lenient over time
        adaptive_accuracy_threshold = base_threshold * iteration_factor
        adaptive_composite_threshold = base_threshold * iteration_factor * 0.3  # Even more lenient
        
        # Check for any meaningful improvement (same logic as complete system)
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
            return True
        else:
            # Only stop if we've tried multiple iterations and see no improvement
            if iteration >= 2:  # Give at least 2 iterations before considering stopping
                print(f"⏹️  No meaningful improvement detected after {iteration} iterations:")
                print(f"     Accuracy improvement: {accuracy_improvement:.3f} (threshold: {adaptive_accuracy_threshold:.3f})")
                print(f"     Composite score: {composite_score:.3f} (threshold: {adaptive_composite_threshold:.3f})")
                print(f"     F1 improvement: {f1_improvement:.3f} (threshold: 0.02)")
                print(f"     Failed cases reduction: {failed_cases_reduction} (threshold: 2)")
                return False
            else:
                print(f"⚠️  Limited improvement in iteration {iteration}, but continuing to explore...")
                print(f"     Will reassess after iteration {iteration + 1}")
                return True
    
    def _convert_results_to_response(
        self,
        request_id: str,
        request: OptimizationRequest,
        results: Dict[str, Any],
        start_time: datetime
    ) -> OptimizationResult:
        """Convert internal results to FastAPI response format"""
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Extract data splits info
        data_splits = results['data_splits']
        data_split_summary = DataSplitSummary(
            total_samples=len(data_splits['train']) + len(data_splits['dev_a']) + len(data_splits['dev_b']) + len(data_splits['test']),
            train_samples=len(data_splits['train']),
            dev_a_samples=len(data_splits['dev_a']),
            dev_b_samples=len(data_splits['dev_b']),
            test_samples=len(data_splits['test'])
        )
        
        # Extract optimization results
        optimization_results = results['optimization_results']
        best_candidate = optimization_results.get('best_candidate', {})
        
        # Build iterations history
        iterations_history = []
        for iter_result in optimization_results.get('optimization_history', []):
            candidate = iter_result['candidate']
            candidate_prompt = (candidate.get('candidate_prompt') or 
                              candidate.get('prompt') or 
                              candidate.get('optimized_prompt') or 
                              'Unknown prompt')
            iterations_history.append(OptimizationIterationResult(
                iteration=iter_result['iteration'],
                optimizer_used=candidate.get('optimizer_used', 'freeform'),
                candidate_prompt=candidate_prompt,
                dev_a_metrics=candidate['dev_a_metrics'],
                improvement_over_baseline=candidate['improvement_over_baseline'],
                reasoning=candidate.get('reasoning', ''),
                confidence=candidate.get('confidence', 0.5)
            ))
        
        # Enhanced human feedback summary (same format as complete system)
        human_feedback_summary = None
        if best_candidate and best_candidate.get('human_feedback_summary'):
            hf_summary = best_candidate['human_feedback_summary']
            # Handle both dataclass and dict formats
            if hasattr(hf_summary, 'total_cases'):  # dataclass format
                human_feedback_summary = HumanFeedbackSummary(
                    total_cases_reviewed=hf_summary.total_cases,
                    accuracy_improvement=0.0,  # Calculate from feedback
                    key_insights=hf_summary.key_feedback_themes[:3] if hf_summary.key_feedback_themes else [],
                    problematic_fields=hf_summary.most_problematic_fields[:3] if hf_summary.most_problematic_fields else []
                )
            else:  # dict format (fallback)
                human_feedback_summary = HumanFeedbackSummary(
                    total_cases_reviewed=hf_summary.get('total_cases_reviewed', 0),
                    accuracy_improvement=hf_summary.get('accuracy_improvement', 0.0),
                    key_insights=hf_summary.get('key_insights', []),
                    problematic_fields=hf_summary.get('problematic_fields', [])
                )
        
        # Run final evaluation and deployment decision (same as complete system)
        final_improvement, deployment_recommendation = self._make_deployment_decision(
            optimization_results, results['baseline_metrics'], results['test_metrics']
        )
        
        # Save final prompt with deployment decision
        final_prompt = optimization_results.get('recommended_prompt', results['baseline_system_prompt'])
        asyncio.create_task(self._save_final_prompt(
            request_id, final_prompt, deployment_recommendation, final_improvement
        ))
        
        return OptimizationResult(
            request_id=request_id,
            status="completed",
            data_split=data_split_summary,
            baseline_prompt=f"{results['baseline_system_prompt']}\n\nUser Query Template: {results['user_prompt_template']}",
            baseline_metrics=results['baseline_metrics'],
            total_iterations=optimization_results.get('total_iterations', 0),
            iterations_history=iterations_history,
            best_prompt=final_prompt,
            best_metrics=best_candidate.get('dev_a_metrics', {}) if best_candidate else {},
            improvement_percentage=final_improvement * 100,
            human_feedback_summary=human_feedback_summary,
            test_metrics=results['test_metrics'],
            deployment_recommendation=deployment_recommendation,
            total_execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _make_deployment_decision(
        self,
        optimization_results: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Make deployment decision based on comprehensive evaluation (same logic as complete system)
        """
        best_candidate = optimization_results.get('best_candidate', {})
        
        if not best_candidate:
            return 0.0, "baseline"
        
        # Calculate comprehensive improvements on test data
        baseline_accuracy = baseline_metrics['overall_accuracy']
        test_accuracy = test_metrics['overall_accuracy']
        accuracy_improvement = test_accuracy - baseline_accuracy
        
        baseline_f1 = baseline_metrics['summary']['average_enum_macro_f1']
        test_f1 = test_metrics['summary']['average_enum_macro_f1']
        f1_improvement = test_f1 - baseline_f1
        
        baseline_valid_json = baseline_metrics['validation_metrics']['valid_json_accuracy']
        test_valid_json = test_metrics['validation_metrics']['valid_json_accuracy']
        valid_json_improvement = test_valid_json - baseline_valid_json
        
        baseline_failed_cases = len(baseline_metrics['detailed_failed_cases']['wrong_classifications'])
        test_failed_cases = len(test_metrics['detailed_failed_cases']['wrong_classifications'])
        failed_cases_reduction = baseline_failed_cases - test_failed_cases
        
        # Calculate comprehensive score (same weights as complete system)
        comprehensive_score = (
            accuracy_improvement * 3.0 +      # Primary metric (weight: 3)
            f1_improvement * 2.0 +            # F1 score (weight: 2)
            valid_json_improvement * 1.5 +    # JSON validity (weight: 1.5)
            (failed_cases_reduction / max(baseline_failed_cases, 1)) * 1.0  # Failed cases reduction (weight: 1)
        )
        
        # Enhanced deployment decision based on comprehensive score (same logic as complete system)
        if comprehensive_score > 0.01:  # Positive comprehensive improvement
            deployment_decision = "deploy"
            print("✅ RECOMMENDATION: Deploy optimized prompt (comprehensive improvement)")
        elif accuracy_improvement > 0.005:  # Small but positive accuracy improvement
            deployment_decision = "deploy"
            print("✅ RECOMMENDATION: Deploy optimized prompt (accuracy improvement)")
        elif f1_improvement > 0.02 and valid_json_improvement >= 0:  # Good F1 improvement with no JSON regression
            deployment_decision = "deploy"
            print("✅ RECOMMENDATION: Deploy optimized prompt (F1 improvement)")
        elif failed_cases_reduction > 0 and accuracy_improvement >= -0.01:  # Fewer failures with minimal accuracy loss
            deployment_decision = "deploy"
            print("✅ RECOMMENDATION: Deploy optimized prompt (fewer failed cases)")
        else:
            deployment_decision = "baseline"
            print("⚠️  RECOMMENDATION: Keep baseline prompt (no significant improvement)")
        
        print(f"\n📈 Final Test Results:")
        print(f"   Accuracy Improvement: {accuracy_improvement:+.3f}")
        print(f"   F1 Score Improvement: {f1_improvement:+.3f}")
        print(f"   Valid JSON Improvement: {valid_json_improvement:+.3f}")
        print(f"   Failed Cases Reduction: {failed_cases_reduction:+d}")
        print(f"   Comprehensive Score: {comprehensive_score:+.3f}")
        
        return comprehensive_score, deployment_decision
    
    async def _update_progress(
        self,
        request_id: str,
        step: str,
        progress: float,
        message: str,
        current_iteration: Optional[int] = None,
        total_iterations: Optional[int] = None,
        estimated_time_remaining: Optional[int] = None
    ):
        """Update optimization progress"""
        if request_id in self._active_optimizations:
            self._active_optimizations[request_id].update({
                "current_step": step,
                "progress_percentage": progress,
                "message": message,
                "current_iteration": current_iteration,
                "total_iterations": total_iterations,
                "estimated_time_remaining": estimated_time_remaining
            })
        
        # Optional: Log progress
        print(f"[{request_id}] {step}: {progress:.1f}% - {message}")
    
    def _create_enhanced_system_prompt(
        self, 
        system_prompt: str, 
        schema: Dict[str, Any]
    ) -> str:
        """
        Create enhanced system prompt with schema and instructions
        Only the system prompt gets optimized - user prompt stays separate as query template
        """
        import json
        
        # Start with the system prompt only
        enhanced_prompt = system_prompt
        
        # Add schema information
        schema_section = "\n\nThe json schema with the fields and their possible values (enum values) is as follows:\n"
        schema_section += json.dumps(schema, indent=2)
        
        # Add important instructions
        instructions = "\n\nIMPORTANT: Respond with a valid JSON object only. The value should be from enum values only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON."
        
        return enhanced_prompt + schema_section + instructions
    
    async def cleanup_optimization(self, request_id: str):
        """Clean up optimization resources"""
        if request_id in self._active_optimizations:
            del self._active_optimizations[request_id] 
    
    async def _create_intermediate_results_dir(self, request_id: str):
        """Create intermediate results directory structure"""
        try:
            base_dir = f"intermediate_results/{request_id}"
            os.makedirs(base_dir, exist_ok=True)
            print(f"💾 Created intermediate results directory: {base_dir}")
        except Exception as e:
            print(f"⚠️  Failed to create intermediate results directory: {e}")
    
    async def _copy_baseline_files_to_request_folder(self, request_id: str):
        """Initialize request-specific folder structure (no copying needed - fresh results per request)"""
        try:
            request_dir = f"intermediate_results/{request_id}"
            # Directory already created by _create_intermediate_results_dir
            print(f"💾 Request folder ready: {request_dir}")
            print(f"📝 All baseline files will be generated fresh for this request")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize request folder: {e}")
    
    async def _save_baseline_prompt(self, request_id: str, enhanced_system_prompt: str, user_prompt_template: str, schema: Dict[str, Any]):
        """Save the baseline prompt to intermediate results folder"""
        try:
            baseline_dir = f"intermediate_results/{request_id}/baseline"
            os.makedirs(baseline_dir, exist_ok=True)
            
            # Save baseline prompt metadata
            baseline_data = {
                "timestamp": datetime.now().isoformat(),
                "prompt_type": "baseline",
                "schema": schema,
                "enhanced_system_prompt": enhanced_system_prompt,
                "user_prompt_template": user_prompt_template
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
                f.write(f"Schema: {json.dumps(schema, indent=2)}\n")
                f.write("\nENHANCED SYSTEM PROMPT:\n")
                f.write("=" * 80 + "\n")
                f.write(enhanced_system_prompt)
                f.write("\n\nUSER PROMPT TEMPLATE:\n")
                f.write("=" * 80 + "\n")
                f.write(user_prompt_template)
            
            print(f"💾 Saved baseline prompt to: {baseline_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save baseline prompt: {e}")
    
    async def _save_data_splits(self, request_id: str, data_splits: Dict[str, List]):
        """Save data splits to intermediate results folder"""
        try:
            data_file = f"intermediate_results/{request_id}/data_splits.json"
            with open(data_file, "w") as f:
                json.dump(data_splits, f, indent=2, default=str)
            print(f"💾 Saved data splits to: {data_file}")
        except Exception as e:
            print(f"⚠️  Failed to save data splits: {e}")
    
    async def _save_train_baseline_metrics(self, request_id: str, train_baseline_metrics: Dict[str, Any]):
        """Save train baseline metrics to intermediate results folder"""
        try:
            metrics_file = f"intermediate_results/{request_id}/train_baseline_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(train_baseline_metrics, f, indent=2, default=str)
            print(f"💾 Saved train baseline metrics to: {metrics_file}")
        except Exception as e:
            print(f"⚠️  Failed to save train baseline metrics: {e}")
    
    async def _save_enhanced_baseline_results(self, request_id: str, train_baseline_metrics: Dict[str, Any]):
        """Save baseline metrics in the format expected by existing components"""
        try:
            metrics_file = f"intermediate_results/{request_id}/enhanced_baseline_results.json"
            with open(metrics_file, "w") as f:
                json.dump(train_baseline_metrics, f, indent=2, default=str)
            print(f"💾 Saved enhanced baseline results to: {metrics_file}")
        except Exception as e:
            print(f"⚠️  Failed to save enhanced baseline results: {e}")
    
    async def _save_dev_a_baseline_metrics(self, request_id: str, dev_a_baseline_metrics: Dict[str, Any]):
        """Save dev_a baseline metrics to intermediate results folder"""
        try:
            metrics_file = f"intermediate_results/{request_id}/dev_a_baseline_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(dev_a_baseline_metrics, f, indent=2, default=str)
            print(f"💾 Saved dev_a baseline metrics to: {metrics_file}")
        except Exception as e:
            print(f"⚠️  Failed to save dev_a baseline metrics: {e}")
    
    async def _save_intent_analysis(self, request_id: str, intent_analysis: Dict[str, Any]):
        """Save intent analysis to intermediate results folder"""
        try:
            analysis_file = f"intermediate_results/{request_id}/intent_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(intent_analysis, f, indent=2, default=str)
            print(f"💾 Saved intent analysis to: {analysis_file}")
        except Exception as e:
            print(f"⚠️  Failed to save intent analysis: {e}")
    
    async def _save_candidate_prompts(self, request_id: str, candidates: List[Dict[str, Any]], iteration: int):
        """Save all generated candidate prompts to intermediate results folder"""
        try:
            candidates_dir = f"intermediate_results/{request_id}/iteration_{iteration:02d}_candidates"
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
                    optimized_prompt = (candidate.get('optimized_prompt') or 
                                      candidate.get('candidate_prompt') or 
                                      candidate.get('prompt') or 
                                      'No prompt found')
                    f.write(optimized_prompt)
            
            print(f"💾 Saved {len(candidates)} candidate prompts to: {candidates_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save candidate prompts: {e}")
    
    async def _save_selected_prompt(self, request_id: str, best_candidate: Dict[str, Any], iteration: int):
        """Save the selected best prompt to intermediate results folder"""
        try:
            selected_dir = f"intermediate_results/{request_id}/iteration_{iteration:02d}_selected"
            os.makedirs(selected_dir, exist_ok=True)
            
            # Save selected prompt metadata
            selected_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "strategy": best_candidate.get('strategy', 'Unknown'),
                "confidence": best_candidate.get('confidence', 0.0),
                "dev_a_improvement": best_candidate.get('improvement_over_baseline', 0.0),
                "composite_score": best_candidate.get('composite_score', 0.0),
                "f1_improvement": best_candidate.get('f1_improvement', 0.0),
                "valid_json_improvement": best_candidate.get('valid_json_improvement', 0.0),
                "failed_cases_reduction": best_candidate.get('failed_cases_reduction', 0),
                "execution_time": best_candidate.get('execution_time', 0.0),
                "changes_made": best_candidate.get('changes_made', []),
                "reasoning": best_candidate.get('reasoning', ''),
                "dev_a_metrics": best_candidate.get('dev_a_metrics', {}),
                "optimized_prompt": (best_candidate.get('optimized_prompt') or 
                                   best_candidate.get('candidate_prompt') or 
                                   best_candidate.get('prompt') or 
                                   'No prompt found')
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
                f.write(f"Dev A Improvement: {best_candidate.get('improvement_over_baseline', 0.0):+.3f}\n")
                f.write(f"Composite Score: {best_candidate.get('composite_score', 0.0):+.3f}\n")
                f.write(f"F1 Improvement: {best_candidate.get('f1_improvement', 0.0):+.3f}\n")
                f.write(f"Valid JSON Improvement: {best_candidate.get('valid_json_improvement', 0.0):+.3f}\n")
                f.write(f"Failed Cases Reduction: {best_candidate.get('failed_cases_reduction', 0):+d}\n")
                f.write(f"Execution Time: {best_candidate.get('execution_time', 0.0):.2f}s\n")
                f.write(f"Changes Made: {', '.join(best_candidate.get('changes_made', []))}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("\nReasoning:\n")
                f.write("-" * 40 + "\n")
                f.write(best_candidate.get('reasoning', 'No reasoning provided'))
                f.write("\n\n")
                f.write("OPTIMIZED PROMPT:\n")
                f.write("=" * 80 + "\n")
                optimized_prompt = (best_candidate.get('optimized_prompt') or 
                                  best_candidate.get('candidate_prompt') or 
                                  best_candidate.get('prompt') or 
                                  'No prompt found')
                f.write(optimized_prompt)
            
            print(f"💾 Saved selected prompt to: {selected_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save selected prompt: {e}")
    
    async def _save_optimization_results(self, request_id: str, optimization_results: Dict[str, Any]):
        """Save optimization results to intermediate results folder"""
        try:
            results_file = f"intermediate_results/{request_id}/optimization_results.json"
            with open(results_file, "w") as f:
                json.dump(optimization_results, f, indent=2, default=str)
            print(f"💾 Saved optimization results to: {results_file}")
        except Exception as e:
            print(f"⚠️  Failed to save optimization results: {e}")
    
    async def _save_test_metrics(self, request_id: str, test_metrics: Dict[str, Any]):
        """Save test metrics to intermediate results folder"""
        try:
            metrics_file = f"intermediate_results/{request_id}/test_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(test_metrics, f, indent=2, default=str)
            print(f"💾 Saved test metrics to: {metrics_file}")
        except Exception as e:
            print(f"⚠️  Failed to save test metrics: {e}")
    
    async def _save_final_prompt(self, request_id: str, recommended_prompt: str, deployment_decision: str, comprehensive_score: float):
        """Save the final recommended prompt to intermediate results folder"""
        try:
            final_dir = f"intermediate_results/{request_id}/final"
            os.makedirs(final_dir, exist_ok=True)
            
            # Save final prompt metadata
            final_data = {
                "timestamp": datetime.now().isoformat(),
                "deployment_decision": deployment_decision,
                "comprehensive_score": comprehensive_score,
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
                f.write(f"Comprehensive Score: {comprehensive_score:+.3f}\n")
                f.write("\nRECOMMENDED PROMPT:\n")
                f.write("=" * 80 + "\n")
                f.write(recommended_prompt)
            
            print(f"💾 Saved final recommended prompt to: {final_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to save final prompt: {e}") 