"""
Context Manager - Smart context assembly and management for optimization
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import copy
import sys
import os
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# Add parent directory to path for imports when running standalone
from prompt_optimizer.models.types import (
    OptimizationContext, 
    EvaluationMetrics, 
    PromptHistory,
    ModelConfiguration
)

from prompt_optimizer.utils.claude_client import ClaudeClient, ClaudeAPIError
from prompt_optimizer.core.claude_intent_identifier import load_baseline_metrics
from prompt_optimizer.core.request_id import get_request_id


class ContextManager:
    """
    Smart context manager for optimization process
    Handles assembly, updates, history management, and compression
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client
        self._context_history: List[OptimizationContext] = []
        self._max_history_size = 10  # Keep last 10 contexts
        self._max_failed_cases = 5  # Limit failed cases for performance
        self._max_human_feedback = 5  # Keep last 5 feedback items
    
    def create_initial_context(
        self,
        json_schema: Dict[str, Any],
        failed_cases_summary: Dict[str, Any],
        failed_cases: Dict[str, List[Dict[str, Any]]],
        baseline_metrics: Dict[str, Any],
        intent: Dict[str, Any],
        base_prompt: str,
        target_model: ModelConfiguration
    ) -> OptimizationContext:
        """
        Create the initial optimization context with model information
        """
        print("create_initial_context")
        
        # Extract failed cases list from the dictionary structure
        # failed_cases comes as: {'invalid_json': [], 'schema_violations': [], 'wrong_classifications': [...]}
        # Take up to 3 examples from each field that has data
        failed_cases_list = []
        max_examples_per_field = 3
        
        print(f"Processing failed cases from {len(failed_cases)} failure types:")
        
        # Process each failure type
        for failure_type, cases in failed_cases.items():
            if isinstance(cases, list) and len(cases) > 0:
                # Take up to 3 examples from this field
                examples_to_take = min(max_examples_per_field, len(cases))
                selected_cases = cases[:examples_to_take]  # Take first N cases
                
                print(f"  - {failure_type}: {len(cases)} total, taking {examples_to_take} examples")
                
                # Add failure_type to each case for context
                for case in selected_cases:
                    case_with_type = case.copy() if isinstance(case, dict) else case
                    if isinstance(case_with_type, dict):
                        case_with_type['failure_type'] = failure_type
                    failed_cases_list.append(case_with_type)
            else:
                print(f"  - {failure_type}: empty or not a list, skipping")
        
        print(f"Total failed cases extracted: {len(failed_cases_list)}")
        
        # Create EvaluationMetrics from raw data
        eval_metrics = EvaluationMetrics(
            baseline_metrics=baseline_metrics,
            failed_cases=failed_cases_list,  # Now using the list format
            failed_cases_summary=failed_cases_summary,
            evaluated_with=target_model 
        )
        
        print(f"Failed cases: {len(failed_cases_list)} total cases extracted from {len(failed_cases)} failure types (max {max_examples_per_field} per type)")
        
        # Create initial context
        context = OptimizationContext(
            json_schema=json_schema,
            failed_cases_summary=failed_cases_summary,
            failed_cases=failed_cases_list,  # Use the extracted list
            baseline_metrics=baseline_metrics,
            intent=intent,
            base_prompt=base_prompt,
            target_model=target_model,
            prompt_history=[],
            human_feedback=[],
            iteration_number=1,
        )
        
        # Store in history
        self._add_to_history(context)
        
        print(f"✅ Created initial context.")
        print(f"   Target model: {target_model.to_string()}")
        print(f"   Failed cases: {len(failed_cases_list)} (max {max_examples_per_field} per failure type)")
        print(f"   Iteration: {context.iteration_number}")
        
        return context
    
    def update_context_with_results(
        self,
        current_context: OptimizationContext,
        new_prompt: str,
        new_metrics: Dict[str, Any],
        optimizer_used: str,
        human_feedback: Optional[str] = None,
        human_feedback_summary: Optional[Dict[str, Any]] = None,
        new_failed_cases: Optional[List[Dict[str, Any]]] = None,
        new_failed_cases_summary: Optional[Dict[str, Any]] = None
    ) -> OptimizationContext:
        """
        Update context with new optimization results
        Model information is preserved from the current context
        """
        
        # Create new evaluation metrics with model information
        new_eval_metrics = EvaluationMetrics(
            baseline_metrics=new_metrics,
            failed_cases=new_failed_cases or [],
            failed_cases_summary=new_failed_cases_summary or {},
            evaluated_with=current_context.target_model  # Same model as we're optimizing for
        )
        
        # Create new prompt history entry
        new_history_entry = PromptHistory(
            prompt=new_prompt,
            metrics=new_eval_metrics,
            timestamp=datetime.now(),
            iteration=current_context.iteration_number,
            optimizer_used=optimizer_used,
            tested_with=current_context.target_model  # Track which model this was tested with
        )
        
        # Update prompt history (keep reasonable size)
        updated_history = current_context.prompt_history + [new_history_entry]
        if len(updated_history) > 3:  # Keep last 3 attempts
            updated_history = updated_history[-3:]
        
        # Update human feedback (keep recent ones) with smart summarization
        updated_feedback = current_context.human_feedback.copy()
        if human_feedback:
            updated_feedback.append(human_feedback)
        
        # Add summarized human feedback if available (much more efficient than raw data)
        if human_feedback_summary:
            # Convert feedback summary to concise text format for context
            summary_text = self._format_feedback_summary(human_feedback_summary)
            updated_feedback.append(summary_text)
        
        if len(updated_feedback) > self._max_human_feedback:
            updated_feedback = updated_feedback[-self._max_human_feedback:]
        
        # Update failed cases if provided
        updated_failed_cases = current_context.failed_cases
        updated_failed_cases_summary = current_context.failed_cases_summary
        if new_failed_cases:
            updated_failed_cases = new_failed_cases[:self._max_failed_cases]
            updated_failed_cases_summary = new_failed_cases_summary or current_context.failed_cases_summary
        
        # Create updated context - new metrics become the baseline (natural insights)
        updated_context = OptimizationContext(
            intent=current_context.intent,
            base_prompt=new_prompt,  # Update to new prompt
            json_schema=current_context.json_schema,
            target_model=current_context.target_model,  # Preserve target model
            baseline_metrics=new_metrics,  # Updated metrics become new baseline (Dict, not EvaluationMetrics)
            failed_cases=updated_failed_cases,  # List[Dict], not EvaluationMetrics
            failed_cases_summary=updated_failed_cases_summary,  # Dict, not EvaluationMetrics
            prompt_history=updated_history,
            human_feedback=updated_feedback,
            iteration_number=current_context.iteration_number + 1
        )
        
        # Store in history
        self._add_to_history(updated_context)
        
        print(f"✅ Updated context - Iteration {updated_context.iteration_number}")
        print(f"   Target model: {updated_context.target_model.to_string()}")
        print(f"   New prompt length: {len(new_prompt)} chars")
        print(f"   History entries: {len(updated_history)}")
        print(f"   Human feedback items: {len(updated_feedback)}")
        
        return updated_context
    
    def merge_contexts(
        self,
        primary_context: OptimizationContext,
        additional_data: Dict[str, Any]
    ) -> OptimizationContext:
        """
        Smart merge of additional data into existing context
        Note: dev_b_insights removed - updated metrics naturally become insights
        """
        
        # Deep copy to avoid modifying original
        merged_context = copy.deepcopy(primary_context)
        
        # Merge additional failed cases
        if "failed_cases" in additional_data:
            existing_cases = merged_context.failed_cases  # This is already a List[Dict]
            new_cases = additional_data["failed_cases"]
            
            # Combine and deduplicate (simple string comparison)
            all_cases = existing_cases + new_cases
            unique_cases = []
            seen_inputs = set()
            
            for case in all_cases:
                case_input = case.get("input", "")
                if case_input not in seen_inputs:
                    unique_cases.append(case)
                    seen_inputs.add(case_input)
            
            # Limit size and update
            merged_context.failed_cases = unique_cases[:self._max_failed_cases]
        
        # Merge human feedback
        if "human_feedback" in additional_data:
            new_feedback = additional_data["human_feedback"]
            if isinstance(new_feedback, str):
                new_feedback = [new_feedback]
            
            merged_feedback = merged_context.human_feedback + new_feedback
            # Keep recent feedback only
            if len(merged_feedback) > self._max_human_feedback:
                merged_feedback = merged_feedback[-self._max_human_feedback:]
            merged_context.human_feedback = merged_feedback
        
        print(f"✅ Merged additional data into context")
        print(f"   Failed cases: {len(merged_context.failed_cases)}")
        print(f"   Human feedback: {len(merged_context.human_feedback)}")
        
        return merged_context
    
    def get_context_summary(self, context: OptimizationContext) -> Dict[str, Any]:
        """
        Get a concise summary of context for optimization
        """
        
        # Extract key accuracy metric
        accuracy = self._extract_accuracy(context.baseline_metrics)  # baseline_metrics is already a Dict
        
        return {
            "intent": context.intent,
            "target_model": context.target_model.to_string(),
            "current_accuracy": accuracy,
            "failed_cases_count": len(context.failed_cases),  # failed_cases is already a List
            "iterations_attempted": len(context.prompt_history),
            "has_human_feedback": len(context.human_feedback) > 0,
            "optimization_iteration": context.iteration_number
        }
    
    def _extract_accuracy(self, metrics: Dict[str, Any]) -> float:
        """Extract accuracy from metrics dict"""
        if "accuracy" in metrics:
            return float(metrics["accuracy"])
        elif "acc" in metrics:
            return float(metrics["acc"])
        else:
            # Fallback calculation if accuracy not directly available
            return 0.0
    
    def _add_to_history(self, context: OptimizationContext):
        """Add context to history with size management"""
        self._context_history.append(context)
        if len(self._context_history) > self._max_history_size:
            self._context_history = self._context_history[-self._max_history_size:]
    
    def get_context_history(self) -> List[OptimizationContext]:
        """Get context history"""
        return self._context_history.copy()
    
    def _format_feedback_summary(self, feedback_summary: Dict[str, Any]) -> str:
        """
        Format feedback summary into concise text for context management
        This avoids passing all raw human feedback data while preserving insights
        """
        if hasattr(feedback_summary, 'total_cases'):
            # Handle dataclass format
            total = feedback_summary.total_cases
            correct = feedback_summary.correct_count
            incorrect = feedback_summary.incorrect_count
            skipped = feedback_summary.skipped_count
            confidence = feedback_summary.average_confidence
            themes = feedback_summary.key_feedback_themes
            suggestions = feedback_summary.improvement_suggestions
        else:
            # Handle dict format
            total = feedback_summary.get('total_cases', 0)
            correct = feedback_summary.get('correct_count', 0)
            incorrect = feedback_summary.get('incorrect_count', 0)
            skipped = feedback_summary.get('skipped_count', 0)
            confidence = feedback_summary.get('average_confidence', 0.0)
            themes = feedback_summary.get('key_feedback_themes', [])
            suggestions = feedback_summary.get('improvement_suggestions', [])
        
        summary_text = f"HUMAN_FEEDBACK_SUMMARY: {total} cases reviewed - "
        summary_text += f"Correct: {correct}, Incorrect: {incorrect}, Skipped: {skipped}. "
        summary_text += f"Reviewer confidence: {confidence:.2f}. "
        
        if themes:
            summary_text += f"Key issues: {', '.join(themes[:2])}. "
        
        if suggestions:
            summary_text += f"Main suggestion: {suggestions[0][:100]}..."
        
        return summary_text
    
    async def close(self):
        """Cleanup resources"""
        if self.claude_client:
            await self.claude_client.close() 
            
if __name__ == "__main__":
    
    # These should be loaded from the request-specific intermediate results folder
    request_id = get_request_id() or "default"
    all_metrics = load_baseline_metrics(f"fastapi_optimization_system/intermediate_results/{request_id}/enhanced_baseline_results.json")
    with open(f"fastapi_optimization_system/intermediate_results/{request_id}/claude_intent_analysis_results.json", 'r', encoding='utf-8') as f:
        intent_analysis = json.load(f)
    base_prompt = """
    You are an expert code generation classifier. Analyze the user's request and classify it according to the provided schema.
    Return your response as a valid JSON object with the specified fields.
    
    Create a page for jpeg to png image converter.
    
    IMPORTANT: Respond with a valid JSON object only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON.
    """
    
    context_manager = ContextManager()
    context = context_manager.create_initial_context(
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
    
    print("="*60)
    print("Context history:")
    print("="*60)
    for context in context_manager.get_context_history():
        print(f"Intent: {context.intent}")
        print(f"Target model: {context.target_model.to_string()}")
        print(f"Failed cases: {len(context.failed_cases)}")
        print(f"Iterations attempted: {len(context.prompt_history)}")
        print(f"Has human feedback: {len(context.human_feedback) > 0}")
        print(f"Optimization iteration: {context.iteration_number}")
        print("="*60)