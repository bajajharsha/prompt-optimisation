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

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from complete_optimization_system.data_manager import DataManager
from complete_optimization_system.evaluation_engine import EvaluationEngine
from complete_optimization_system.optimization_controller import OptimizationController
from complete_optimization_system.human_feedback_integration import HumanFeedbackIntegration

class CompleteOptimizationSystem:
    """
    Main system that orchestrates the complete optimization workflow
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.evaluation_engine = EvaluationEngine()
        self.optimization_controller = OptimizationController()
        self.human_feedback = HumanFeedbackIntegration()
        
        # Configuration
        self.config = {
            "max_iterations": 5,
            "improvement_threshold": 0.05,  # 5% improvement to continue
            "dataset_name": "code_gen",
            "target_model": {
                "provider": "groq",
                "model_name": "llama-3.3-70b-versatile"
            }
        }
        
        # Schema definition
        self.schema = {
            "action": ["CODE_GENERATION", "NOT_FOUND"],
            "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
            "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", 
                        "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
            "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
            "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
        }
    
    async def run_complete_optimization(self) -> Dict[str, Any]:
        """
        Run the complete optimization workflow
        """
        print("🚀 Starting Complete Prompt Optimization System")
        print("=" * 80)
        
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
            
            # Step 2: Baseline Evaluation
            print("\n📈 Step 2: Baseline Evaluation")
            baseline_prompt = self._get_baseline_prompt()
            
            baseline_metrics = await self.evaluation_engine.evaluate_prompt(
                prompt=baseline_prompt,
                data=data_splits['dev_a'],
                schema=self.schema,
                evaluation_type="baseline"
            )
            
            print(f"✅ Baseline evaluation completed:")
            print(f"   Overall Accuracy: {baseline_metrics['overall_accuracy']:.3f}")
            print(f"   Average F1: {baseline_metrics['summary']['average_enum_macro_f1']:.3f}")
            print(f"   Failed Cases: {len(baseline_metrics['detailed_failed_cases']['wrong_classifications'])}")
            
            # Step 3: Intent Analysis
            print("\n🎯 Step 3: Intent Analysis")
            intent_analysis = await self.optimization_controller.analyze_intent(
                schema=self.schema,
                baseline_metrics=baseline_metrics,
                train_samples=data_splits['train'][:5],  # Use 5 train samples for context
                base_prompt=baseline_prompt
            )
            
            print("✅ Intent analysis completed")
            
            # Step 4: Optimization Loop
            print("\n🔄 Step 4: Optimization Loop")
            optimization_results = await self._run_optimization_loop(
                baseline_prompt=baseline_prompt,
                baseline_metrics=baseline_metrics,
                intent_analysis=intent_analysis,
                data_splits=data_splits
            )
            
            # Step 5: Final Test Evaluation
            print("\n🧪 Step 5: Final Test Evaluation")
            final_results = await self._run_final_evaluation(
                optimization_results=optimization_results,
                baseline_metrics=baseline_metrics,
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
        baseline_metrics: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        data_splits: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Run the iterative optimization loop
        """
        current_prompt = baseline_prompt
        current_metrics = baseline_metrics
        iteration = 1
        optimization_history = []
        
        while iteration <= self.config["max_iterations"]:
            print(f"\n🔄 Optimization Iteration {iteration}")
            print("-" * 50)
            
            # Generate candidate prompts
            print("📝 Generating candidate prompts...")
            candidates = await self.optimization_controller.generate_candidates(
                current_prompt=current_prompt,
                current_metrics=current_metrics,
                intent_analysis=intent_analysis,
                schema=self.schema,
                iteration=iteration
            )
            
            if not candidates:
                print("❌ No candidates generated, stopping optimization")
                break
            
            print(f"✅ Generated {len(candidates)} candidate prompts")
            
            # Evaluate candidates on Dev A
            print("📊 Evaluating candidates on Dev A...")
            best_candidate = await self._evaluate_candidates_dev_a(
                candidates=candidates,
                dev_a_data=data_splits['dev_a'],
                baseline_metrics=current_metrics
            )
            
            if not best_candidate:
                print("❌ No improved candidates found, stopping optimization")
                break
            
            print(f"✅ Best candidate selected: {best_candidate['strategy']}")
            print(f"   Dev A Improvement: {best_candidate['improvement']:.3f}")
            
            # Check stopping criteria (based on Dev A)
            if best_candidate['improvement'] < self.config["improvement_threshold"]:
                print(f"⏹️  Improvement below threshold ({self.config['improvement_threshold']:.3f}), stopping")
                break
            
            # Evaluate on Dev B for human feedback
            print("🔬 Evaluating on Dev B...")
            dev_b_results = await self.evaluation_engine.evaluate_prompt(
                prompt=best_candidate['optimized_prompt'],
                data=data_splits['dev_b'],
                schema=self.schema,
                evaluation_type="dev_b_validation"
            )
            
            print(f"✅ Dev B evaluation completed:")
            print(f"   Overall Accuracy: {dev_b_results['overall_accuracy']:.3f}")
            print(f"   Average F1: {dev_b_results['summary']['average_enum_macro_f1']:.3f}")
            
            # Collect human feedback
            print("👥 Collecting human feedback...")
            human_feedback_results = await self.human_feedback.collect_feedback(
                candidate_prompt=best_candidate['optimized_prompt'],
                dev_b_results=dev_b_results,
                baseline_metrics=baseline_metrics
            )
            
            # Update for next iteration
            current_prompt = best_candidate['optimized_prompt']
            current_metrics = dev_b_results
            
            # Store iteration results
            iteration_result = {
                "iteration": iteration,
                "strategy": best_candidate['strategy'],
                "dev_a_improvement": best_candidate['improvement'],
                "dev_b_metrics": dev_b_results,
                "human_feedback": human_feedback_results,
                "optimized_prompt": best_candidate['optimized_prompt']
            }
            optimization_history.append(iteration_result)
            
            print(f"✅ Iteration {iteration} completed")
            iteration += 1
        
        return {
            "final_prompt": current_prompt,
            "final_metrics": current_metrics,
            "optimization_history": optimization_history,
            "total_iterations": iteration - 1
        }
    
    async def _evaluate_candidates_dev_a(
        self,
        candidates: List[Dict[str, Any]],
        dev_a_data: List[Dict[str, Any]],
        baseline_metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate all candidates on Dev A and return the best one
        """
        best_candidate = None
        best_improvement = 0.0
        baseline_accuracy = baseline_metrics['overall_accuracy']
        
        for candidate in candidates:
            print(f"   Evaluating: {candidate['strategy']}")
            
            # Evaluate candidate
            candidate_metrics = await self.evaluation_engine.evaluate_prompt(
                prompt=candidate['optimized_prompt'],
                data=dev_a_data,
                schema=self.schema,
                evaluation_type="dev_a_candidate"
            )
            
            # Calculate improvement
            improvement = candidate_metrics['overall_accuracy'] - baseline_accuracy
            
            print(f"     Accuracy: {candidate_metrics['overall_accuracy']:.3f} (Δ{improvement:+.3f})")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = {
                    **candidate,
                    'dev_a_metrics': candidate_metrics,
                    'improvement': improvement
                }
        
        return best_candidate
    
    async def _run_final_evaluation(
        self,
        optimization_results: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run final evaluation on test data and make deployment decision
        """
        final_prompt = optimization_results['final_prompt']
        
        # Evaluate final prompt on test data
        print("📊 Evaluating final prompt on test data...")
        test_metrics = await self.evaluation_engine.evaluate_prompt(
            prompt=final_prompt,
            data=test_data,
            schema=self.schema,
            evaluation_type="final_test"
        )
        
        # Evaluate baseline on test data for comparison
        print("📊 Evaluating baseline prompt on test data...")
        baseline_test_metrics = await self.evaluation_engine.evaluate_prompt(
            prompt=self._get_baseline_prompt(),
            data=test_data,
            schema=self.schema,
            evaluation_type="baseline_test"
        )
        
        # Compare and make decision
        test_improvement = test_metrics['overall_accuracy'] - baseline_test_metrics['overall_accuracy']
        
        print(f"\n📈 Final Test Results:")
        print(f"   Baseline Test Accuracy: {baseline_test_metrics['overall_accuracy']:.3f}")
        print(f"   Optimized Test Accuracy: {test_metrics['overall_accuracy']:.3f}")
        print(f"   Test Improvement: {test_improvement:+.3f}")
        
        # Deployment decision
        if test_improvement > 0:
            deployment_decision = "DEPLOY"
            recommended_prompt = final_prompt
            print("✅ RECOMMENDATION: Deploy optimized prompt")
        else:
            deployment_decision = "KEEP_BASELINE"
            recommended_prompt = self._get_baseline_prompt()
            print("⚠️  RECOMMENDATION: Keep baseline prompt")
        
        return {
            "deployment_decision": deployment_decision,
            "recommended_prompt": recommended_prompt,
            "test_improvement": test_improvement,
            "baseline_test_metrics": baseline_test_metrics,
            "optimized_test_metrics": test_metrics,
            "optimization_results": optimization_results
        }
    
    def _get_baseline_prompt(self) -> str:
        """Get the baseline prompt"""
        return """You are a classification model. Classify the input into the correct category. Return the result in JSON format.

The schema is as follows:
{
"action": ["CODE_GENERATION", "NOT_FOUND"],
"subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
"platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
"framework": ["REACT", "FLUTTER", "NOT_FOUND"],
"languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
}

IMPORTANT: Respond with a valid JSON object only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON."""


async def main():
    """Main entry point"""
    system = CompleteOptimizationSystem()
    
    try:
        results = await system.run_complete_optimization()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"complete_optimization_system/results/optimization_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print final summary
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Decision: {results['deployment_decision']}")
        print(f"   Test Improvement: {results['test_improvement']:+.3f}")
        print(f"   Total Iterations: {results['optimization_results']['total_iterations']}")
        
    except Exception as e:
        print(f"❌ System failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 