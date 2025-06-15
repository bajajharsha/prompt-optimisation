import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import asyncio
from pathlib import Path

# Add the parent directory to sys.path to import existing components
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from complete_optimization_system.main import CompleteOptimizationSystem
from complete_optimization_system.data_manager import DataManager
from complete_optimization_system.evaluation_engine import EvaluationEngine
from complete_optimization_system.optimization_controller import OptimizationController
from complete_optimization_system.human_feedback_integration import HumanFeedbackIntegration

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
            from complete_optimization_system.request_id import initialize_request_id
            initialize_request_id()
            
            # Set up the enhanced baseline prompt with schema and instructions
            baseline_prompt = self._create_enhanced_baseline_prompt(
                config['system_prompt'], 
                config['user_prompt'], 
                config['schema']
            )
            
            # Update progress
            await self._update_progress(request_id, "data_loading", 20.0, "Loading and splitting dataset...")
            
            # Use existing data manager
            data_manager = DataManager()
            data_splits = await data_manager.prepare_data_splits(config['dataset'])
            
            # Update progress
            await self._update_progress(request_id, "baseline_evaluation", 30.0, "Evaluating baseline performance...")
            
            # Use existing evaluation engine
            evaluation_engine = EvaluationEngine()
            
            # Evaluate baseline on train data
            train_baseline_metrics = await evaluation_engine.evaluate_prompt(
                baseline_prompt,
                data_splits['train'],
                config['schema'],
                "train_baseline"
            )
            
            # Evaluate baseline on dev_a (hidden comparison dataset)
            dev_a_baseline_metrics = await evaluation_engine.evaluate_prompt(
                baseline_prompt,
                data_splits['dev_a'],
                config['schema'],
                "dev_a_baseline"
            )
            
            # Update progress
            await self._update_progress(request_id, "intent_analysis", 40.0, "Analyzing optimization intent...")
            
            # Use existing optimization controller for intent analysis
            optimization_controller = OptimizationController()
            intent_analysis = await optimization_controller.analyze_intent(
                config['schema'],
                train_baseline_metrics,
                data_splits['train'][:5],  # Sample for intent analysis
                baseline_prompt
            )
            
            # Update progress
            await self._update_progress(request_id, "optimization_loop", 50.0, "Starting optimization iterations...")
            
            # Run the optimization loop
            optimization_results = await self._run_optimization_loop(
                request_id,
                baseline_prompt,
                train_baseline_metrics,
                dev_a_baseline_metrics,
                intent_analysis,
                data_splits,
                config
            )
            
            # Update progress
            await self._update_progress(request_id, "final_evaluation", 90.0, "Running final evaluation...")
            
            # Final evaluation on test data
            final_prompt = optimization_results.get('recommended_prompt', baseline_prompt)
            test_metrics = await evaluation_engine.evaluate_prompt(
                final_prompt,
                data_splits['test'],
                config['schema'],
                "test_final"
            )
            
            # Update progress
            await self._update_progress(request_id, "completed", 100.0, "Optimization completed!")
            
            return {
                "data_splits": data_splits,
                "baseline_metrics": train_baseline_metrics,
                "dev_a_baseline_metrics": dev_a_baseline_metrics,
                "intent_analysis": intent_analysis,
                "optimization_results": optimization_results,
                "test_metrics": test_metrics,
                "baseline_prompt": baseline_prompt
            }
            
        except Exception as e:
            raise OptimizationProcessError(
                f"Optimization process failed: {str(e)}",
                request_id=request_id
            )
    
    async def _run_optimization_loop(
        self,
        request_id: str,
        baseline_prompt: str,
        train_baseline_metrics: Dict[str, Any],
        dev_a_baseline_metrics: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        data_splits: Dict[str, List],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the optimization iteration loop"""
        
        optimization_controller = OptimizationController()
        evaluation_engine = EvaluationEngine()
        human_feedback_integration = HumanFeedbackIntegration()
        
        current_prompt = baseline_prompt
        current_metrics = train_baseline_metrics
        optimization_history = []
        best_candidate = None
        
        max_iterations = config.get('max_iterations', 5)
        improvement_threshold = config.get('improvement_threshold', 0.05)
        
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
                
                # Generate candidate prompts
                candidates = await optimization_controller.generate_candidates(
                    current_prompt,
                    current_metrics,
                    intent_analysis,
                    config['schema'],
                    iteration,
                    optimization_history=optimization_history
                )
                
                if not candidates:
                    print(f"No candidates generated for iteration {iteration}")
                    break
                
                # Evaluate candidates on dev_a
                best_iteration_candidate = None
                best_iteration_improvement = -1.0
                
                for candidate in candidates:
                    candidate_metrics = await evaluation_engine.evaluate_prompt(
                        candidate['candidate_prompt'],
                        data_splits['dev_a'],
                        config['schema'],
                        f"dev_a_iteration_{iteration}"
                    )
                    
                    # Compare with baseline
                    comparison = evaluation_engine.compare_metrics(
                        dev_a_baseline_metrics,
                        candidate_metrics
                    )
                    
                    improvement = comparison.get('overall_improvement', 0.0)
                    candidate['dev_a_metrics'] = candidate_metrics
                    candidate['improvement_over_baseline'] = improvement
                    
                    if improvement > best_iteration_improvement:
                        best_iteration_improvement = improvement
                        best_iteration_candidate = candidate
                
                if best_iteration_candidate is None:
                    print(f"No suitable candidate found in iteration {iteration}")
                    break
                
                # Check if we should continue optimization
                if best_iteration_improvement < improvement_threshold:
                    print(f"Improvement {best_iteration_improvement} below threshold {improvement_threshold}")
                    break
                
                # Run candidate on dev_b for human feedback
                dev_b_metrics = await evaluation_engine.evaluate_prompt(
                    best_iteration_candidate['candidate_prompt'],
                    data_splits['dev_b'],
                    config['schema'],
                    f"dev_b_iteration_{iteration}"
                )
                
                # Collect human feedback if enabled
                human_feedback_summary = None
                if config.get('enable_human_feedback', True):
                    try:
                        human_feedback_data = await human_feedback_integration.collect_feedback(
                            best_iteration_candidate['candidate_prompt'],
                            dev_b_metrics,
                            dev_a_baseline_metrics
                        )
                        human_feedback_summary = human_feedback_data
                    except Exception as e:
                        print(f"Human feedback collection failed: {e}")
                
                # Update optimization history
                iteration_result = {
                    'iteration': iteration,
                    'candidate': best_iteration_candidate,
                    'dev_b_metrics': dev_b_metrics,
                    'human_feedback': human_feedback_summary,
                    'improvement': best_iteration_improvement
                }
                optimization_history.append(iteration_result)
                
                # Update current best
                if best_candidate is None or best_iteration_improvement > best_candidate.get('improvement_over_baseline', 0):
                    best_candidate = best_iteration_candidate.copy()
                    best_candidate['iteration'] = iteration
                    best_candidate['dev_b_metrics'] = dev_b_metrics
                    best_candidate['human_feedback'] = human_feedback_summary
                
                # Update for next iteration
                current_prompt = best_iteration_candidate['candidate_prompt']
                current_metrics = best_iteration_candidate['dev_a_metrics']
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                break
        
        return {
            'optimization_history': optimization_history,
            'best_candidate': best_candidate,
            'total_iterations': len(optimization_history),
            'recommended_prompt': best_candidate['candidate_prompt'] if best_candidate else baseline_prompt,
            'final_improvement': best_candidate.get('improvement_over_baseline', 0.0) if best_candidate else 0.0
        }
    
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
            iterations_history.append(OptimizationIterationResult(
                iteration=iter_result['iteration'],
                optimizer_used=candidate.get('optimizer_used', 'freeform'),
                candidate_prompt=candidate['candidate_prompt'],
                dev_a_metrics=candidate['dev_a_metrics'],
                improvement_over_baseline=candidate['improvement_over_baseline'],
                reasoning=candidate.get('reasoning', ''),
                confidence=candidate.get('confidence', 0.5)
            ))
        
        # Human feedback summary
        human_feedback_summary = None
        if best_candidate and best_candidate.get('human_feedback'):
            hf_data = best_candidate['human_feedback']
            human_feedback_summary = HumanFeedbackSummary(
                total_cases_reviewed=hf_data.get('total_cases_reviewed', 0),
                accuracy_improvement=hf_data.get('accuracy_improvement', 0.0),
                key_insights=hf_data.get('key_insights', []),
                problematic_fields=hf_data.get('problematic_fields', [])
            )
        
        # Determine deployment recommendation
        final_improvement = optimization_results.get('final_improvement', 0.0)
        deployment_recommendation = "deploy" if final_improvement > request.improvement_threshold else "baseline"
        
        return OptimizationResult(
            request_id=request_id,
            status="completed",
            data_split=data_split_summary,
            baseline_prompt=results['baseline_prompt'],
            baseline_metrics=results['baseline_metrics'],
            total_iterations=optimization_results.get('total_iterations', 0),
            iterations_history=iterations_history,
            best_prompt=optimization_results.get('recommended_prompt', results['baseline_prompt']),
            best_metrics=best_candidate.get('dev_a_metrics', {}) if best_candidate else {},
            improvement_percentage=final_improvement * 100,
            human_feedback_summary=human_feedback_summary,
            test_metrics=results['test_metrics'],
            deployment_recommendation=deployment_recommendation,
            total_execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
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
    
    def _create_enhanced_baseline_prompt(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        schema: Dict[str, Any]
    ) -> str:
        """
        Create enhanced baseline prompt with schema and instructions
        Matches the format used in complete_optimization_system
        """
        import json
        
        # Start with system and user prompts
        base_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Add schema information
        schema_section = "\n\nThe json schema with the fields and their possible values (enum values) is as follows:\n"
        schema_section += json.dumps(schema, indent=2)
        
        # Add important instructions
        instructions = "\n\nIMPORTANT: Respond with a valid JSON object only. The value should be from enum values only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON."
        
        return base_prompt + schema_section + instructions
    
    async def cleanup_optimization(self, request_id: str):
        """Clean up optimization resources"""
        if request_id in self._active_optimizations:
            del self._active_optimizations[request_id] 