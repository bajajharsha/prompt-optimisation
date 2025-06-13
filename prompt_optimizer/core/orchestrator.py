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


