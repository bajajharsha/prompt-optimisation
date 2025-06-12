"""
Freeform Optimizer - Intelligent prompt optimization using Claude
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

from .base_optimizer import BaseOptimizer
from ..models.types import OptimizationContext, OptimizerResult, OptimizationStatus
from ..utils.claude_client import ClaudeClient, ClaudeAPIError


class FreeformOptimizer(BaseOptimizer):
    """
    Intelligent freeform prompt optimizer using Claude
    
    Analyzes context, failed cases, and target model to create optimized prompts
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        super().__init__()
        self.claude_client = claude_client or ClaudeClient()
        self.optimizer_name = "freeform"
    
    async def optimize(self, context: OptimizationContext) -> OptimizerResult:
        """
        Optimize prompt using intelligent analysis of context and target model
        """
        start_time = time.time()
        
        try:
            # Build comprehensive context message including model information
            context_message = self._build_context_message(context)
            
            # Get optimization from Claude
            optimization_response = await self.claude_client.complete(
                message=context_message,
                component="freeform_optimizer",
                operation="prompt_optimization"
            )
            
            # Parse the JSON response
            try:
                result_data = json.loads(optimization_response.strip())
                
                optimized_prompt = result_data.get("optimized_prompt", context.base_prompt)
                reasoning = result_data.get("reasoning", "No reasoning provided")
                confidence = float(result_data.get("confidence", 0.5))
                changes_made = result_data.get("changes_made", [])
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Failed to parse optimization response: {e}")
                # Fallback: return original prompt
                optimized_prompt = context.base_prompt
                reasoning = f"Failed to parse optimization response: {str(e)}"
                confidence = 0.1
                changes_made = ["Parsing failed - returned original prompt"]
            
            execution_time = time.time() - start_time
            
            return OptimizerResult(
                optimizer_name=self.optimizer_name,
                candidate_prompt=optimized_prompt,
                reasoning=reasoning,
                confidence=confidence,
                changes_made=changes_made,
                execution_time=execution_time,
                optimized_for=context.target_model,  # Track which model this was optimized for
                timestamp=datetime.now(),
                status=OptimizationStatus.COMPLETED
            )
            
        except ClaudeAPIError as e:
            execution_time = time.time() - start_time
            print(f"❌ Claude API error in freeform optimizer: {e}")
            
            return OptimizerResult(
                optimizer_name=self.optimizer_name,
                candidate_prompt=context.base_prompt,  # Return original on error
                reasoning=f"Claude API error: {str(e)}",
                confidence=0.0,
                changes_made=[],
                execution_time=execution_time,
                optimized_for=context.target_model,
                timestamp=datetime.now(),
                status=OptimizationStatus.FAILED,
                error_message=str(e)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Unexpected error in freeform optimizer: {e}")
            
            return OptimizerResult(
                optimizer_name=self.optimizer_name,
                candidate_prompt=context.base_prompt,
                reasoning=f"Unexpected error: {str(e)}",
                confidence=0.0,
                changes_made=[],
                execution_time=execution_time,
                optimized_for=context.target_model,
                timestamp=datetime.now(),
                status=OptimizationStatus.FAILED,
                error_message=str(e)
            )
    
    def _build_context_message(self, context: OptimizationContext) -> str:
        """
        Build comprehensive context message including model-specific information
        """
        
        # Extract key metrics
        baseline_metrics = context.baseline_metrics.baseline_metrics
        failed_cases = context.failed_cases.failed_cases[:5]  # Limit for message size
        
        # Build model-specific guidance
        model_guidance = self._get_model_specific_guidance(context.target_model)
        
        # Build failed cases summary
        failed_cases_text = ""
        if failed_cases:
            failed_cases_text = "\n".join([
                f"- Input: '{case.get('input_text', '')}' | Expected: {case.get('expected_intent', '')} | Got: {case.get('predicted_intent', '')} | Confidence: {case.get('confidence', 'N/A')}"
                for case in failed_cases
            ])
        
        # Build optimization history context
        history_context = ""
        if context.prompt_history:
            recent_attempts = context.prompt_history[-3:]  # Last 3 attempts
            history_context = "\n\nPrevious optimization attempts:\n"
            for i, attempt in enumerate(recent_attempts, 1):
                history_context += f"{i}. {attempt.optimizer_used or 'Unknown'} optimizer - {attempt.iteration} iterations ago\n"
        
        # Build human feedback context
        feedback_context = ""
        if context.human_feedback:
            recent_feedback = context.human_feedback[-3:]  # Last 3 feedback items
            feedback_context = "\n\nRecent human feedback:\n"
            feedback_context += "\n".join([f"- {feedback}" for feedback in recent_feedback])
        
        message = f"""You are an expert prompt optimization system. Your task is to improve the given prompt for better performance on the target model.

**TARGET MODEL INFORMATION:**
- Provider: {context.target_model.provider}
- Model: {context.target_model.model_name}
- Temperature: {context.target_model.temperature}
{model_guidance}

**TASK CONTEXT:**
Intent: {context.intent}
Current Accuracy: {baseline_metrics.get('accuracy', 'Unknown'):.2%}
Iteration: {context.iteration_number}

**CURRENT PROMPT:**
{context.base_prompt}

**JSON SCHEMA REQUIREMENT:**
{json.dumps(context.json_schema, indent=2)}

**FAILED CASES ANALYSIS:**
{failed_cases_text if failed_cases_text else "No specific failed cases provided"}

**PERFORMANCE METRICS:**
- Accuracy: {baseline_metrics.get('accuracy', 'N/A')}
- Precision: {baseline_metrics.get('precision', 'N/A')}
- Recall: {baseline_metrics.get('recall', 'N/A')}
- F1-Score: {baseline_metrics.get('f1_score', 'N/A')}
{history_context}
{feedback_context}

**OPTIMIZATION INSTRUCTIONS:**
1. Analyze the failed cases to identify patterns in misclassification
2. Consider the target model's strengths and optimal prompt format
3. Make the prompt more generalizable and robust to edge cases
4. Ensure the prompt works well with the specified JSON schema
5. Maintain consistency with the task intent while improving accuracy

Provide your optimization as a JSON response with this structure:
{{
    "optimized_prompt": "Your improved prompt here",
    "reasoning": "Detailed explanation of what you changed and why",
    "confidence": 0.8,
    "changes_made": ["List of specific changes made"]
}}

Focus on making prompts that are:
- Clear and unambiguous for the target model
- Robust to edge cases and variations
- Optimized for the specific model architecture and training
- Consistent with the JSON schema requirements"""
        
        return message
    
    def _get_model_specific_guidance(self, model_config) -> str:
        """
        Get model-specific optimization guidance
        """
        provider = model_config.provider.lower()
        model_name = model_config.model_name.lower()
        
        guidance = ""
        
        if provider == "anthropic":
            guidance += "\n- Claude models work well with clear, structured instructions"
            guidance += "\n- Use explicit formatting and step-by-step reasoning"
            guidance += "\n- Provide clear examples when beneficial"
            
            if "sonnet" in model_name:
                guidance += "\n- Sonnet excels at complex reasoning and detailed analysis"
                guidance += "\n- Can handle longer, more detailed prompts effectively"
            elif "haiku" in model_name:
                guidance += "\n- Haiku is optimized for speed and conciseness"
                guidance += "\n- Keep prompts clear but concise"
        
        elif provider == "openai":
            guidance += "\n- GPT models respond well to role-based prompting"
            guidance += "\n- Use 'You are a...' style instructions"
            guidance += "\n- Structured examples often improve performance"
            
            if "gpt-4" in model_name:
                guidance += "\n- GPT-4 can handle complex, nuanced instructions"
                guidance += "\n- Benefits from detailed context and examples"
        
        elif provider == "google":
            guidance += "\n- Gemini models work well with structured formats"
            guidance += "\n- Clear task decomposition often helps"
        
        return guidance
    
    def get_name(self) -> str:
        """Return the optimizer name"""
        return self.optimizer_name
    
    def get_description(self) -> str:
        """Return a description of this optimizer"""
        return "Intelligent freeform prompt optimizer that analyzes context, failed cases, and target model characteristics to create optimized prompts" 