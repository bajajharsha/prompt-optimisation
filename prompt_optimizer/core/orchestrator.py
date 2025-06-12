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
            analysis_message = self._build_analysis_message(context)
            with open("analysis_message.txt", "w") as f:
                f.write(analysis_message)
            # return analysis_message
            
            # Get AI strategy recommendation
            response = await self.claude_client.complete(
                message=analysis_message,
                component="orchestrator",
                operation="strategy_selection"
            )
            
            # Parse the response
            try:
                strategy_data = json.loads(response.strip())
                
                return OptimizerSelection(
                    selected_optimizers=strategy_data.get("selected_optimizers", ["freeform"]),
                    reasoning=strategy_data.get("reasoning", "AI-selected strategy"),
                    execution_mode=strategy_data.get("execution_mode", "parallel"),
                    confidence=float(strategy_data.get("confidence", 0.7))
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Failed to parse strategy response: {e}")
                return self._fallback_strategy_selection(context)
                
        except ClaudeAPIError as e:
            print(f"⚠️ Claude API error in strategy selection: {e}")
            return self._fallback_strategy_selection(context)
        
        except Exception as e:
            print(f"⚠️ Unexpected error in strategy selection: {e}")
            return self._fallback_strategy_selection(context)
    
    def _build_analysis_message(self, context: OptimizationContext) -> str:
        """Build comprehensive context analysis message"""
        
        # Get available optimizers for the message
        available_optimizers = get_optimizer_names()
        
        # Extract key metrics
        baseline_metrics = context.baseline_metrics.baseline_metrics if hasattr(context.baseline_metrics, 'baseline_metrics') else {}
        failed_cases = context.failed_cases.failed_cases if hasattr(context.failed_cases, 'failed_cases') else []
        
        # Build failed cases summary
        failed_cases_summary = ""
        if failed_cases:
            failed_cases_summary = "\n".join([
                f"- Input: '{case.get('input_text', '')}' | Expected: {case.get('expected_intent', '')} | Got: {case.get('predicted_intent', '')}"
                for case in failed_cases[:3]  # Show first 3
            ])
        
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
Current baseline performance metrics: {baseline_metrics}
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
{', '.join(available_optimizers)}

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
        {
            "name": "field_focus",
            "reasoning": "High failure rate in category field suggests focused improvement needed"
        },
        {
            "name": "example_enhancement", 
            "reasoning": "Limited examples causing confusion in edge cases"
        }
    ],
    "overall_reasoning": "Detailed explanation of why these optimizers were selected",
}}

Focus on optimizers that can address the specific issues shown in the failed cases."""
        
        return message
    
    async def close(self):
        """Close any resources"""
        if self.claude_client:
            await self.claude_client.close() 
            
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


