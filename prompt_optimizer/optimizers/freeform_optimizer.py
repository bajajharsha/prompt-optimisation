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
        super().__init__(
            name="freeform",
            description="Intelligent freeform prompt optimizer using Claude to analyze context and improve prompts"
        )
        self.claude_client = claude_client
    
    async def optimize(self, context: OptimizationContext) -> OptimizerResult:
        """
        Optimize prompt using intelligent analysis of context and target model
        """
        start_time = time.time()
        
        try:
            # Build comprehensive context message including model information
            context_message = self._build_context_message(context)
            
            with open("freeform_context.md", "w") as f:
                f.write(context_message)
            
            # Get optimization from Claude
            optimization_response = await self.claude_client.complete(
                messages=context_message,
                component="freeform_optimizer",
                operation="prompt_optimization"
            )
            
            with open("freeform_response.md", "w") as f:
                f.write(json.dumps(optimization_response, indent=2))
            
            # Parse the JSON response with robust error handling
            try:
                # Handle the response format: {'content': '```json\n{...}\n```', 'usage': {...}}
                if isinstance(optimization_response, dict) and 'content' in optimization_response:
                    content = optimization_response['content']
                else:
                    content = optimization_response
                
                print(f"🔍 Raw Claude response content (first 500 chars): {content[:500]}")
                
                # Extract JSON from markdown code blocks if present
                json_content = content.strip()
                if '```json' in content:
                    # Find the JSON content between ```json and ```
                    start = content.find('```json') + 7  # Skip ```json
                    end = content.find('```', start)
                    if end != -1:
                        json_content = content[start:end].strip()
                    else:
                        # No closing ```, try to find the JSON object
                        json_start = content.find('{', start)
                        if json_start != -1:
                            # Find the last } in the content
                            json_end = content.rfind('}')
                            if json_end != -1:
                                json_content = content[json_start:json_end + 1]
                
                print(f"🔍 Extracted JSON content (first 300 chars): {json_content[:300]}")
                
                # Try to fix common JSON issues
                json_content = self._fix_common_json_issues(json_content)
                
                # Parse the JSON - handle the specific format we're getting
                result_data = json.loads(json_content)
                
                optimized_prompt = result_data.get("optimized_prompt", context.base_prompt)
                reasoning = result_data.get("reasoning", "No reasoning provided")
                confidence = float(result_data.get("confidence", 0.5))
                changes_made = result_data.get("changes_made", [])
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Failed to parse optimization response: {e}")
                print(f"🔍 Problematic JSON content: {json_content[:1000] if 'json_content' in locals() else 'Not extracted'}")
                
                # Try to extract at least the optimized_prompt using regex
                optimized_prompt = self._extract_prompt_fallback(content, context.base_prompt)
                reasoning = f"Failed to parse optimization response: {str(e)}. Used fallback extraction."
                confidence = 0.1
                changes_made = ["Parsing failed - used fallback extraction"]
            
            execution_time = time.time() - start_time
            
            return OptimizerResult(
                optimizer_name=self.name,
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
                optimizer_name=self.name,
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
                optimizer_name=self.name,
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
        baseline_metrics = context.baseline_metrics
        failed_cases = context.failed_cases[:5]
        
        # Get model information for context (no specific guidance)
        model_info = self._get_model_info_context(context.target_model)
        
        # Build failed cases summary with better field handling
        failed_cases_text = ""
        if failed_cases:
            failed_cases_examples = []
            for case in failed_cases:
                # Handle different case formats
                input_text = case.get('input_text') or case.get('input_prompt') or case.get('input', '')
                expected = case.get('expected_intent') or case.get('expected_output') or case.get('ground_truth', '')
                predicted = case.get('predicted_intent') or case.get('model_output') or case.get('prediction_text', '')
                
                # Format the case for better analysis
                case_text = f"- Input: '{input_text}'"
                if isinstance(expected, dict):
                    case_text += f" | Expected: {json.dumps(expected)}"
                else:
                    case_text += f" | Expected: {expected}"
                    
                if isinstance(predicted, dict):
                    case_text += f" | Got: {json.dumps(predicted)}"
                else:
                    case_text += f" | Got: {predicted}"
                
                # Add wrong fields if available
                if 'wrong_fields' in case:
                    wrong_fields = case['wrong_fields']
                    if wrong_fields:
                        case_text += f" | Issues: {[f['field'] for f in wrong_fields if isinstance(f, dict) and 'field' in f]}"
                
                failed_cases_examples.append(case_text)
            
            failed_cases_text = "\n".join(failed_cases_examples)
        
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

**TASK CONTEXT:**
Intent: {context.intent}
Current Metrics: {context.baseline_metrics}
Iteration: {context.iteration_number}

**CURRENT PROMPT:**
{context.base_prompt}

**JSON SCHEMA REQUIREMENT:**
{json.dumps(context.json_schema, indent=2)}

**FAILED CASES ANALYSIS:**
{failed_cases_text if failed_cases_text else "No specific failed cases provided"}

**HISTORY AND FEEDBACK:**
{history_context}
{feedback_context}

**OPTIMIZATION INSTRUCTIONS:**
You MUST make significant improvements to the prompt. Analyze the failed cases carefully and make aggressive changes:

1. **CRITICAL ANALYSIS**: Look at each failed case and identify exactly why the model made the wrong prediction
2. **PATTERN IDENTIFICATION**: Find common patterns in the failures (e.g., specific fields being misclassified)
3. **AGGRESSIVE OPTIMIZATION**: Make substantial changes to the prompt structure, not just minor tweaks
4. **SPECIFIC GUIDANCE**: Add explicit instructions for handling the types of cases that are failing
5. **SCHEMA ENFORCEMENT**: Ensure the model strictly follows the JSON schema format
6. **EDGE CASE HANDLING**: Add specific instructions for ambiguous or edge cases
7. **NOT STRICT STATEMENTS**: Do not add strict statements according to the failed cases
8. **INTENT MATCHING**: The optimized prompt MUST be more aligned with the intent of the task
9. **JSON SCHEMA**: On the basis of sample data, the schema is provided. The optimized prompt MUST be more aligned with the schema.

**REQUIRED CHANGES:**
- Add explicit examples of correct classifications for problematic cases but not strict statements according to the failed cases
- Include specific instructions for each field in the schema but not strict statements according to the failed cases
- Add clear decision-making criteria for ambiguous inputs
- Restructure the prompt for better clarity and specificity
- Add validation instructions to ensure JSON format compliance

**FAILED CASE FOCUS:**
Pay special attention to the failed cases provided. For each pattern you see:
- Add specific instructions to handle that type of input but not strict statements according to the failed cases
- Include examples that demonstrate the correct classification
- Add decision-making criteria to avoid similar mistakes

Provide your optimization as a JSON response with this structure:
{{
    "optimized_prompt": "Your significantly improved prompt here - MUST be substantially different from the original and should only return the prompt with no extra text or comments",
    "reasoning": "Detailed explanation of the major changes you made and why they address the specific failed cases",
    "confidence": 0.8,
    "changes_made": ["List of specific major changes made - should be substantial improvements"]
}}

**REQUIREMENTS:**
- The optimized prompt MUST be significantly better than the original
- MUST address the specific patterns seen in failed cases
- MUST include explicit examples and decision criteria but not strict statements according to the failed cases
- MUST be more structured and comprehensive than the original
- MUST include field-specific guidance for the JSON schema
"""
        
        return message
    
    def _fix_common_json_issues(self, json_content: str) -> str:
        """Fix common JSON formatting issues"""
        import re
        
        # Remove any trailing commas before closing braces/brackets
        json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
        
        # Fix unterminated strings by adding closing quotes
        # This is a simple heuristic - look for lines that start with a quote but don't end with one
        lines = json_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            # If line starts with a quote but doesn't end with quote or comma, try to fix it
            if (stripped.startswith('"') and 
                not stripped.endswith('"') and 
                not stripped.endswith('",') and
                not stripped.endswith('",')):
                # Add closing quote if it seems to be missing
                if ':' in stripped:
                    # This looks like a key-value pair
                    parts = stripped.split(':', 1)
                    if len(parts) == 2:
                        key_part = parts[0].strip()
                        value_part = parts[1].strip()
                        if key_part.startswith('"') and not key_part.endswith('"'):
                            key_part += '"'
                        if value_part.startswith('"') and not value_part.endswith('"') and not value_part.endswith('",'):
                            value_part += '"'
                        line = f"  {key_part}: {value_part}"
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _extract_prompt_fallback(self, content: str, original_prompt: str) -> str:
        """Extract optimized prompt using regex fallback when JSON parsing fails"""
        import re
        
        # Try to find optimized_prompt value using regex
        patterns = [
            r'"optimized_prompt"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',  # Standard JSON string
            r'"optimized_prompt"\s*:\s*"([^"]*)',  # Unterminated string
            r'optimized_prompt["\s]*:\s*["\s]*([^"]*)',  # Loose matching
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1)
                # Clean up the extracted text
                extracted = extracted.replace('\\"', '"')  # Unescape quotes
                extracted = extracted.strip()
                if len(extracted) > 50:  # Only use if it looks substantial
                    print(f"✅ Extracted prompt using fallback: {extracted[:100]}...")
                    return extracted
        
        print("⚠️ Fallback extraction failed, returning original prompt")
        return original_prompt

    def _get_model_info_context(self, model_config) -> str:
        """
        Get model information as context (no prescriptive guidance)
        """
        provider = model_config.provider.lower()
        model_name = model_config.model_name.lower()
        
        # Just provide factual context about the model
        info = f"\n- Model Type: {provider.title()} {model_name}"
        
        return info
    