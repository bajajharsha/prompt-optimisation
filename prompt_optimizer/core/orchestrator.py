"""
Orchestrator Agent - AI-powered strategy selection for prompt optimization
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.models.types import OptimizationContext, OptimizerSelection, ModelConfiguration
from prompt_optimizer.utils.claude_client import ClaudeClient, ClaudeAPIError
from prompt_optimizer.optimizers.simple_registry import get_available_optimizers, get_optimizer_names
from poc.intent_analysis.claude_intent_identifier import load_baseline_metrics
from .human_feedback import HumanFeedbackManager, DevBHumanFeedbackIntegration

class Orchestrator:
    """
    AI-powered orchestrator that selects optimization strategies
    Uses Claude to analyze context and choose the best optimizers
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client or ClaudeClient()
    
    async def select_optimization_strategy(self, context: OptimizationContext) -> OptimizerSelection:
        """
        Analyze context and select the best optimization strategy using AI
        """
        
        try:
            # Build comprehensive analysis message
            print("select_optimization_strategy")
            analysis_message = self._build_analysis_message(context)
            print("analysis_message", analysis_message)
            with open("analysis_message.txt", "w") as f:
                f.write(analysis_message)
            # return analysis_message
            
            # Get AI strategy recommendation
            response = await self.claude_client.complete(
                messages=analysis_message,
                component="orchestrator",
                operation="strategy_selection"
            )
            
            # Parse the response
            try:
                # Handle the response format: {'content': '```json\n{...}\n```', 'usage': {...}}
                if isinstance(response, dict) and 'content' in response:
                    content = response['content']
                else:
                    content = response
                
                # Extract JSON from markdown code blocks if present
                if '```json' in content:
                    # Find the JSON content between ```json and ```
                    start = content.find('```json') + 7  # Skip ```json
                    end = content.find('```', start)
                    json_content = content[start:end].strip()
                else:
                    json_content = content.strip()
                
                strategy_data = json.loads(json_content)
                
                # Extract optimizer names from the detailed format
                selected_optimizers = []
                optimizer_reasoning = {}
                
                if "selected_optimizers" in strategy_data:
                    for opt in strategy_data["selected_optimizers"]:
                        if isinstance(opt, dict) and "name" in opt:
                            optimizer_name = opt["name"]
                            selected_optimizers.append(optimizer_name)
                            # Store individual reasoning if available
                            if "reasoning" in opt:
                                optimizer_reasoning[optimizer_name] = opt["reasoning"]
                        elif isinstance(opt, str):
                            selected_optimizers.append(opt)
                
                return OptimizerSelection(
                    selected_optimizers=selected_optimizers,
                    overall_reasoning=strategy_data.get("overall_reasoning", strategy_data.get("reasoning", "AI-selected strategy")),
                    strategy_details=strategy_data,
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Failed to parse strategy response: {e}")
                
        except ClaudeAPIError as e:
            print(f"⚠️ Claude API error in strategy selection: {e}")
        
        except Exception as e:
            print(f"⚠️ Unexpected error in strategy selection: {e}")
    
    def _build_analysis_message(self, context: OptimizationContext) -> str:
        """Build comprehensive context analysis message"""
        
        # Get available optimizers for the message
        available_optimizers = get_available_optimizers()
        
        # Build history context
        history_context = ""
        if context.prompt_history:
            recent_attempts = context.prompt_history[-2:]  # Last 2 attempts
            history_context = "\n\nRecent optimization attempts:\n"
            for attempt in recent_attempts:
                history_context += f"- {attempt.optimizer_used}: Iteration {attempt.iteration}\n"
                history_context += f"- Prompt: {attempt.prompt}\n"
        
        # Build human feedback context
        feedback_context = ""
        if context.human_feedback:
            recent_feedback = context.human_feedback[-2:]  # Last 2 feedback items
            feedback_context = "\n\nRecent human feedback:\n"
            feedback_context += "\n".join([f"- {feedback}" for feedback in recent_feedback])
        
        message = f"""You are an expert prompt optimization strategist. Analyze the context and select the best optimizers for this situation.

**CONTEXT ANALYSIS:**
Intent: {context.intent}
Current baseline performance metrics: {context.baseline_metrics}
Target model: {context.target_model.provider}/{context.target_model.model_name}
Optimization iteration: {context.iteration_number}

**CURRENT PROMPT:**
{context.base_prompt}

**FAILED CASES SUMMARY:**
{context.failed_cases_summary}

Failed cases examples:
{context.failed_cases}

History of optimization attempts:
{history_context}

Human feedback:
{feedback_context}

**AVAILABLE OPTIMIZERS:**
{available_optimizers}

**YOUR TASK:**
Based on this analysis, select the most appropriate optimizers and execution strategy. Consider:
1. The current performance issues
2. The failed cases patterns
3. The target model characteristics
4. Previous optimization attempts
5. The complexity of the task

Respond with a JSON object:
{{
    "selected_optimizers": [
        {{
            "name": "optimizer1",
            "reasoning": "Optimizer1 is a good optimizer for this task"
        }},
        {{
            "name": "optimizer2", 
            "reasoning": "Optimizer2 is a good optimizer for this task"
        }}
    ],
    "overall_reasoning": "Detailed explanation of why these optimizers were selected"
}}

Focus on optimizers that can address the specific issues shown in the failed cases."""
        
        return message
    
    async def close(self):
        """Close any resources"""
        if self.claude_client:
            await self.claude_client.close() 
            
    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """
        Run a complete optimization cycle with human feedback integration.
        
        Returns:
            Dict containing optimization results and human feedback
        """
        print("🚀 Starting optimization cycle with human feedback integration...")
        
        try:
            # Step 1: Generate candidate prompts using Dev A
            print("\n📊 Step 1: Generating candidate prompts on Dev A...")
            candidates = await self.generate_candidates()
            
            if not candidates:
                raise ValueError("No candidate prompts generated")
            
            # Step 2: Select best candidate based on Dev A performance
            print(f"\n🎯 Step 2: Selecting best candidate from {len(candidates)} options...")
            best_candidate = self.select_best_candidate(candidates)
            
            print(f"✅ Selected best candidate: {best_candidate['strategy']}")
            print(f"   Dev A F1 Score: {best_candidate.get('dev_a_f1', 'N/A'):.3f}")
            
            # Step 3: Evaluate best candidate on Dev B
            print(f"\n🔬 Step 3: Evaluating best candidate on Dev B...")
            dev_b_results = await self.evaluate_on_dev_b(best_candidate)
            
            # Step 4: Setup and trigger human feedback collection
            print(f"\n👥 Step 4: Setting up human feedback collection...")
            human_feedback_results = await self.collect_human_feedback(
                best_candidate, dev_b_results
            )
            
            # Step 5: Compile final results
            optimization_results = {
                "cycle_id": f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "candidates_generated": len(candidates),
                "best_candidate": best_candidate,
                "dev_b_evaluation": dev_b_results,
                "human_feedback": human_feedback_results,
                "baseline_comparison": self._compare_with_baseline(dev_b_results),
                "recommendations": self._generate_recommendations(
                    best_candidate, dev_b_results, human_feedback_results
                )
            }
            
            print(f"\n✅ Optimization cycle completed successfully!")
            print(f"   Best candidate F1: {dev_b_results.get('summary', {}).get('average_enum_macro_f1', 0):.3f}")
            print(f"   Human feedback collected: {human_feedback_results.get('total_traces', 0)} traces")
            
            return optimization_results
            
        except Exception as e:
            print(f"❌ Error in optimization cycle: {e}")
            raise

    async def collect_human_feedback(self, 
                                   best_candidate: Dict[str, Any], 
                                   dev_b_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect human feedback on Dev B evaluation results.
        
        Args:
            best_candidate: The selected best candidate prompt
            dev_b_results: Results from Dev B evaluation
            
        Returns:
            Dict containing human feedback results
        """
        print("👥 Collecting human feedback on Dev B results...")
        
        try:
            # Initialize human feedback manager
            feedback_manager = HumanFeedbackManager()
            feedback_integration = DevBHumanFeedbackIntegration(feedback_manager)
            
            # Setup human feedback workflow (first time only)
            feedback_integration.setup_human_feedback_workflow()
            
            # Prepare Dev B results for human review
            dev_b_evaluation_data = self._prepare_dev_b_for_human_review(
                best_candidate, dev_b_results
            )
            
            # Create traces for human feedback
            trace_ids = feedback_integration.process_dev_b_results_for_human_feedback(
                dev_b_results=dev_b_evaluation_data,
                candidate_prompt=best_candidate['optimized_prompt'],
                baseline_metrics=self.context_manager.baseline_metrics
            )
            
            print(f"✅ Created {len(trace_ids)} traces for human review")
            print("📝 Manual steps required:")
            print("   1. Go to LangFuse dashboard")
            print("   2. Navigate to 'Annotate' tab")
            print("   3. Add traces to annotation queue")
            print("   4. Complete human annotations")
            
            # Option 1: Wait for human feedback (blocking)
            if self.config.get('wait_for_human_feedback', False):
                print("⏳ Waiting for human feedback...")
                feedback_results = feedback_manager.wait_for_human_feedback(
                    trace_ids=trace_ids,
                    timeout_minutes=self.config.get('human_feedback_timeout_minutes', 60)
                )
            else:
                # Option 2: Return trace IDs for async collection
                feedback_results = {
                    "status": "pending",
                    "trace_ids": trace_ids,
                    "queue_name": feedback_integration.config.queue_name,
                    "instructions": "Complete human annotations in LangFuse dashboard"
                }
            
            return feedback_results
            
        except Exception as e:
            print(f"❌ Error collecting human feedback: {e}")
            return {
                "status": "error",
                "error": str(e),
                "trace_ids": []
            }

    def _prepare_dev_b_for_human_review(self, 
                                      best_candidate: Dict[str, Any], 
                                      dev_b_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare Dev B evaluation results for human review.
        
        Args:
            best_candidate: Selected best candidate
            dev_b_results: Dev B evaluation results
            
        Returns:
            List of evaluation data formatted for human review
        """
        print("📋 Preparing Dev B results for human review...")
        
        evaluation_data = []
        
        # Extract failed cases for human review (most important)
        failed_cases = dev_b_results.get('detailed_failed_cases', {})
        
        # Prioritize wrong classifications for human feedback
        wrong_classifications = failed_cases.get('wrong_classifications', [])
        
        for i, case in enumerate(wrong_classifications):
            evaluation_item = {
                "evaluation_id": f"wrong_classification_{i}",
                "input_prompt": case.get('input_prompt', ''),
                "model_output": case.get('prediction_text', ''),
                "expected_output": case.get('ground_truth', {}),
                "failure_type": "wrong_classification",
                "wrong_fields": case.get('wrong_fields', []),
                "metadata": {
                    "example_idx": case.get('example_idx', i),
                    "timestamp": case.get('timestamp', ''),
                    "candidate_strategy": best_candidate.get('strategy', 'unknown')
                }
            }
            evaluation_data.append(evaluation_item)
        
        # Add some correct cases for comparison (sample)
        # This helps human reviewers understand what good outputs look like
        if len(evaluation_data) < 10:  # Add correct cases if we have few failures
            # We'd need to extract correct cases from the evaluation
            # For now, we'll note this as a TODO
            pass
        
        print(f"✅ Prepared {len(evaluation_data)} cases for human review")
        print(f"   Wrong classifications: {len(wrong_classifications)}")
        
        return evaluation_data

    async def get_human_feedback_results(self, trace_ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve human feedback results for given trace IDs.
        
        Args:
            trace_ids: List of trace IDs to get feedback for
            
        Returns:
            Dict containing human feedback summary
        """
        print(f"📊 Retrieving human feedback for {len(trace_ids)} traces...")
        
        try:
            feedback_manager = HumanFeedbackManager()
            feedback_results = feedback_manager.get_human_feedback_summary(trace_ids)
            
            print(f"✅ Retrieved feedback for {feedback_results['annotated_traces']} traces")
            return feedback_results
            
        except Exception as e:
            print(f"❌ Error retrieving human feedback: {e}")
            return {
                "status": "error",
                "error": str(e),
                "annotated_traces": 0,
                "total_traces": len(trace_ids)
            }

    def _generate_recommendations(self, 
                                best_candidate: Dict[str, Any],
                                dev_b_results: Dict[str, Any], 
                                human_feedback: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on Dev B results and human feedback.
        
        Args:
            best_candidate: Selected best candidate
            dev_b_results: Dev B evaluation results
            human_feedback: Human feedback results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Performance-based recommendations
        dev_b_f1 = dev_b_results.get('summary', {}).get('average_enum_macro_f1', 0)
        baseline_f1 = self.context_manager.baseline_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
        
        if dev_b_f1 > baseline_f1 + 0.05:  # 5% improvement
            recommendations.append(f"✅ Strong improvement: {(dev_b_f1 - baseline_f1)*100:.1f}% F1 gain")
            recommendations.append("🚀 Recommend deploying this optimized prompt")
        elif dev_b_f1 > baseline_f1:
            recommendations.append(f"📈 Modest improvement: {(dev_b_f1 - baseline_f1)*100:.1f}% F1 gain")
            recommendations.append("🤔 Consider further optimization before deployment")
        else:
            recommendations.append(f"📉 Performance regression: {(baseline_f1 - dev_b_f1)*100:.1f}% F1 loss")
            recommendations.append("❌ Do not deploy - continue optimization")
        
        # Human feedback-based recommendations
        if human_feedback.get('status') == 'pending':
            recommendations.append("⏳ Waiting for human feedback - check LangFuse dashboard")
        elif human_feedback.get('annotated_traces', 0) > 0:
            avg_scores = human_feedback.get('average_scores', {})
            
            if 'improvement_over_baseline' in avg_scores:
                improvement_score = avg_scores['improvement_over_baseline'].get('average', 0)
                if improvement_score > 0.7:  # 70% of humans think it's better
                    recommendations.append("👥 Human feedback: Strong preference for optimized prompt")
                elif improvement_score > 0.5:
                    recommendations.append("👥 Human feedback: Moderate preference for optimized prompt")
                else:
                    recommendations.append("👥 Human feedback: Preference for baseline prompt")
            
            if 'classification_accuracy' in avg_scores:
                accuracy_score = avg_scores['classification_accuracy'].get('average', 0)
                if accuracy_score < 0.7:
                    recommendations.append("⚠️  Human feedback indicates accuracy concerns")
        
        # Strategy-specific recommendations
        strategy = best_candidate.get('strategy', '')
        if 'field_specific' in strategy.lower():
            recommendations.append("🎯 Field-specific optimization showed promise - consider expanding")
        elif 'example' in strategy.lower():
            recommendations.append("📚 Example-based optimization effective - consider more examples")
        
        return recommendations

if __name__ == "__main__":
    
    all_metrics = load_baseline_metrics("poc/metrics/enhanced_baseline_results.json")
    with open("poc/intent_analysis/claude_intent_analysis_results.json", 'r', encoding='utf-8') as f:
        intent_analysis = json.load(f)
    base_prompt = """
    You are an expert code generation classifier. Analyze the user's request and classify it according to the provided schema.
    Return your response as a valid JSON object with the specified fields.
    
    Create a page for jpeg to png image converter.
    
    IMPORTANT: Respond with a valid JSON object only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON.
    """
    
    # Create orchestrator without ClaudeClient to avoid API key requirement
    orchestrator = Orchestrator(claude_client=None)
    
    context = OptimizationContext(
        json_schema={
            "action": ["CODE_GENERATION", "NOT_FOUND"],
            "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
            "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", 
                        "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
            "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
            "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
        },
        failed_cases_summary=all_metrics.get("failed_cases_summary", {}),
        failed_cases=all_metrics.get("detailed_failed_cases", {}).get("wrong_classifications", []),
        baseline_metrics={k: v for k, v in all_metrics.items() if k != "detailed_failed_cases"},
        intent=intent_analysis,
        base_prompt=base_prompt,
        target_model=ModelConfiguration(
            provider="groq",
            model_name="llama-3.7-70b-versatile",
        )
    )
    
    # Just test the analysis message creation
    analysis_message = orchestrator._build_analysis_message(context)
    print("Analysis Message:")
    print("=" * 50)
    print(analysis_message)
    print("=" * 50)
    
    # Also write to file as your code already does
    with open("analysis_message.md", "w") as f:
        f.write(analysis_message)
    print("Analysis message saved to analysis_message.md")


