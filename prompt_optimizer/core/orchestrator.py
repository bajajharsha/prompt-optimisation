"""
Orchestrator Agent - AI-powered strategy selection for prompt optimization
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..models.types import (
    OptimizationContext, 
    OptimizerSelection,
    OptimizationStatus
)
from ..utils.claude_client import ClaudeClient, ClaudeAPIError
from ..optimizers.registry import get_global_registry


class OrchestratorAgent:
    """
    AI-powered orchestrator that analyzes context and selects optimization strategies
    Uses Claude to make intelligent decisions about which optimizers to run
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client
        self.optimizer_registry = get_global_registry()
        
        # Strategy selection parameters
        self.max_parallel_optimizers = 3  # Limit parallel execution
        self.min_confidence_threshold = 0.7  # Minimum confidence to select strategy
        
    async def select_optimization_strategies(
        self,
        context: OptimizationContext,
        available_optimizers: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizerSelection:
        """
        Analyze context and select the best optimization strategies
        
        Args:
            context: Current optimization context
            available_optimizers: List of available optimizers (defaults to all registered)
            constraints: Additional constraints (budget, time limits, etc.)
            
        Returns:
            OptimizerSelection with selected strategies and reasoning
        """
        
        print(f"🎯 Orchestrator analyzing context for iteration {context.iteration_number}...")
        
        # Get available optimizers
        if available_optimizers is None:
            available_optimizers = list(self.optimizer_registry.get_all_optimizers().keys())
        
        print(f"   Available optimizers: {', '.join(available_optimizers)}")
        
        try:
            # Check if Claude client is available
            if self.claude_client is None:
                print("⚠️ No Claude client available, using fallback strategy")
                return self._fallback_strategy_selection(available_optimizers)
            
            # Analyze context and select strategies using Claude
            analysis_result = await self._analyze_context_with_claude(
                context, available_optimizers, constraints
            )
            
            # Parse and validate the selection
            selection = self._parse_strategy_selection(analysis_result, available_optimizers)
            
            # Apply constraints and finalize selection
            final_selection = self._apply_constraints(selection, constraints)
            
            print(f"✅ Selected {len(final_selection.selected_optimizers)} strategies:")
            for optimizer in final_selection.selected_optimizers:
                print(f"   • {optimizer}")
            
            return final_selection
            
        except Exception as e:
            print(f"⚠️ Strategy selection failed: {str(e)}")
            print("Falling back to default strategy selection")
            return self._fallback_strategy_selection(available_optimizers)
    
    async def _analyze_context_with_claude(
        self,
        context: OptimizationContext,
        available_optimizers: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use Claude to analyze context and recommend optimization strategies
        """
        
        # Build context analysis prompt
        system_prompt = self._build_analysis_system_prompt(available_optimizers)
        context_message = self._build_context_analysis_message(context, constraints)
        
        # Get Claude's analysis
        response = await self.claude_client.analyze_context(
            system_prompt=system_prompt,
            context_data=context_message,
            component="orchestrator_strategy_selection"
        )
        
        return response
    
    def _build_analysis_system_prompt(self, available_optimizers: List[str]) -> str:
        """Build the system prompt for strategy analysis"""
        
        # Get optimizer descriptions
        optimizer_descriptions = {}
        for name in available_optimizers:
            optimizer_class = self.optimizer_registry.get_optimizer(name)
            if optimizer_class:
                optimizer_descriptions[name] = getattr(optimizer_class, '__doc__', 'No description available')
        
        return f"""You are an expert prompt optimization strategist. Analyze the provided optimization context and recommend the best optimization strategies.

Available Optimizers:
{json.dumps(optimizer_descriptions, indent=2)}

Your task:
1. Analyze the current prompt performance and identify key issues
2. Review failed cases and patterns to understand failure modes
3. Consider optimization history to avoid repeating ineffective approaches
4. Select 1-3 optimizers that are most likely to improve performance
5. Provide detailed reasoning for each selection

Respond with JSON in this exact format:
{{
    "context_analysis": {{
        "current_performance": "Brief assessment of current prompt performance",
        "key_issues": ["List of main issues identified"],
        "failure_patterns": ["Patterns observed in failed cases"],
        "optimization_potential": "Assessment of improvement opportunities"
    }},
    "strategy_recommendations": [
        {{
            "optimizer": "optimizer_name",
            "confidence": 0.85,
            "reasoning": "Detailed explanation for why this optimizer is recommended",
            "expected_improvements": ["List of expected improvements"],
            "priority": 1
        }}
    ],
    "overall_reasoning": "Summary of overall strategy and approach"
}}

Guidelines:
- Select optimizers based on actual context analysis, not randomly
- Confidence should reflect how well the optimizer matches the identified issues
- Priority 1 = highest priority, 2 = medium, 3 = lowest
- Focus on optimizers that address the specific failure patterns observed
- Consider diminishing returns - don't over-optimize if performance is already high"""

    def _build_context_analysis_message(
        self,
        context: OptimizationContext,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the context message for Claude analysis"""
        
        # Extract key metrics
        current_accuracy = self._extract_accuracy(context.baseline_metrics.baseline_metrics)
        failed_cases_count = len(context.failed_cases.failed_cases if hasattr(context.failed_cases, 'failed_cases') else [])
        
        # Build comprehensive context message
        message_parts = [
            f"OPTIMIZATION CONTEXT ANALYSIS",
            f"Intent: {context.intent}",
            f"Current Iteration: {context.iteration_number}",
            f"Current Accuracy: {current_accuracy:.3f}",
            f"",
            f"CURRENT PROMPT:",
            f"```",
            f"{context.base_prompt}",
            f"```",
            f"",
            f"PERFORMANCE METRICS:",
            json.dumps(context.baseline_metrics.baseline_metrics, indent=2),
            f"",
            f"FAILED CASES ({failed_cases_count} total):"
        ]
        
        # Add failed cases (limit for readability)
        failed_cases = context.failed_cases.failed_cases if hasattr(context.failed_cases, 'failed_cases') else []
        for i, case in enumerate(failed_cases[:10]):  # Show first 10 failed cases
            message_parts.append(f"{i+1}. Input: {case.get('input', 'N/A')}")
            message_parts.append(f"   Expected: {case.get('expected', 'N/A')}")
            message_parts.append(f"   Actual: {case.get('actual', 'N/A')}")
            message_parts.append(f"   Error: {case.get('error', 'N/A')}")
            message_parts.append("")
        
        if len(failed_cases) > 10:
            message_parts.append(f"... and {len(failed_cases) - 10} more failed cases")
            message_parts.append("")
        
        # Add failure summary
        message_parts.extend([
            f"FAILURE PATTERNS SUMMARY:",
            json.dumps(context.failed_cases_summary, indent=2),
            f""
        ])
        
        # Add optimization history if available
        if context.prompt_history:
            message_parts.extend([
                f"OPTIMIZATION HISTORY:",
                f"Previous attempts: {len(context.prompt_history)}"
            ])
            
            for i, history_entry in enumerate(context.prompt_history[-3:]):  # Last 3 attempts
                iteration_num = len(context.prompt_history) - 3 + i + 1
                hist_accuracy = self._extract_accuracy(history_entry.metrics.baseline_metrics)
                message_parts.extend([
                    f"  Attempt {iteration_num}: {history_entry.optimizer_used} -> Accuracy: {hist_accuracy:.3f}",
                ])
            message_parts.append("")
        
        # Add human feedback if available
        if context.human_feedback:
            message_parts.extend([
                f"HUMAN FEEDBACK:",
                "\n".join(f"- {feedback}" for feedback in context.human_feedback[-3:]),  # Last 3 feedback items
                f""
            ])
        
        # Add Dev B insights if available
        if context.dev_b_insights:
            message_parts.extend([
                f"DEV B TESTING INSIGHTS:",
                json.dumps(context.dev_b_insights, indent=2),
                f""
            ])
        
        # Add constraints if provided
        if constraints:
            message_parts.extend([
                f"CONSTRAINTS:",
                json.dumps(constraints, indent=2),
                f""
            ])
        
        # Add JSON schema for reference
        message_parts.extend([
            f"TARGET JSON SCHEMA:",
            json.dumps(context.json_schema, indent=2)
        ])
        
        return "\n".join(message_parts)
    
    def _parse_strategy_selection(
        self,
        analysis_result: Dict[str, Any],
        available_optimizers: List[str]
    ) -> OptimizerSelection:
        """Parse Claude's analysis into OptimizerSelection"""
        
        try:
            # Extract the analysis content
            content = analysis_result.get("content", "{}")
            
            # Parse JSON response
            if isinstance(content, str):
                analysis_data = json.loads(content)
            else:
                analysis_data = content
            
            # Extract recommendations
            recommendations = analysis_data.get("strategy_recommendations", [])
            
            # Filter and validate recommendations
            valid_optimizers = []
            reasoning_parts = []
            
            for rec in recommendations:
                optimizer_name = rec.get("optimizer", "")
                confidence = rec.get("confidence", 0.0)
                
                # Validate optimizer exists and meets confidence threshold
                if (optimizer_name in available_optimizers and 
                    confidence >= self.min_confidence_threshold):
                    valid_optimizers.append(optimizer_name)
                    reasoning_parts.append(f"{optimizer_name} (confidence: {confidence:.2f}): {rec.get('reasoning', 'No reasoning provided')}")
            
            # Build comprehensive reasoning
            context_analysis = analysis_data.get("context_analysis", {})
            overall_reasoning = analysis_data.get("overall_reasoning", "No overall reasoning provided")
            
            full_reasoning = f"""Context Analysis:
- Current Performance: {context_analysis.get('current_performance', 'Not analyzed')}
- Key Issues: {', '.join(context_analysis.get('key_issues', []))}
- Failure Patterns: {', '.join(context_analysis.get('failure_patterns', []))}

Strategy Selection:
{chr(10).join(reasoning_parts)}

Overall Strategy: {overall_reasoning}"""
            
            # Limit to max parallel optimizers
            if len(valid_optimizers) > self.max_parallel_optimizers:
                # Sort by confidence and take top N
                sorted_recs = sorted(recommendations, key=lambda x: x.get("confidence", 0), reverse=True)
                valid_optimizers = [r["optimizer"] for r in sorted_recs[:self.max_parallel_optimizers] 
                                 if r["optimizer"] in available_optimizers]
            
            # Calculate overall confidence as average of selected optimizers
            selected_confidences = [
                rec.get("confidence", 0.0) 
                for rec in recommendations 
                if rec["optimizer"] in valid_optimizers
            ]
            overall_confidence = sum(selected_confidences) / len(selected_confidences) if selected_confidences else 0.5
            
            return OptimizerSelection(
                selected_optimizers=valid_optimizers,
                reasoning=full_reasoning,
                confidence=overall_confidence
            )
            
        except Exception as e:
            print(f"⚠️ Failed to parse strategy selection: {str(e)}")
            # Return fallback with available optimizers
            return self._fallback_strategy_selection(available_optimizers)
    
    def _apply_constraints(
        self,
        selection: OptimizerSelection,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizerSelection:
        """Apply constraints to the selection"""
        
        if not constraints:
            return selection
        
        # Apply budget constraints
        if "max_optimizers" in constraints:
            max_opts = constraints["max_optimizers"]
            if len(selection.selected_optimizers) > max_opts:
                # Keep first N optimizers (they should already be sorted by priority)
                selection.selected_optimizers = selection.selected_optimizers[:max_opts]
        
        # Apply time constraints (for future use)
        if "max_time_minutes" in constraints:
            # Could filter out time-intensive optimizers
            pass
        
        # Apply cost constraints (for future use)
        if "max_cost" in constraints:
            # Could filter out expensive optimizers
            pass
        
        return selection
    
    def _fallback_strategy_selection(self, available_optimizers: List[str]) -> OptimizerSelection:
        """Fallback strategy when AI analysis fails"""
        
        print("🔄 Using fallback strategy selection")
        
        # Use simple heuristic: prefer freeform optimizer if available
        selected = []
        if "freeform" in available_optimizers:
            selected.append("freeform")
        elif available_optimizers:
            selected.append(available_optimizers[0])  # Take first available
        
        return OptimizerSelection(
            selected_optimizers=selected,
            reasoning="Fallback selection due to analysis failure. Using default freeform optimizer for general prompt improvement.",
            confidence=0.5  # Moderate confidence for fallback
        )
    
    def _extract_accuracy(self, metrics: Dict[str, Any]) -> float:
        """Extract accuracy from metrics dictionary"""
        # Try common accuracy field names
        for key in ["accuracy", "overall_accuracy", "total_accuracy"]:
            if key in metrics:
                return float(metrics[key])
        
        # Fallback calculation
        if "correct_predictions" in metrics and "total_samples" in metrics:
            return float(metrics["correct_predictions"]) / float(metrics["total_samples"])
        
        return 0.0
    
    async def analyze_optimization_potential(
        self,
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """
        Analyze the optimization potential without selecting strategies
        Useful for understanding context before making decisions
        """
        
        try:
            # Check if Claude client is available
            if self.claude_client is None:
                return {
                    "potential_score": 0.5,
                    "key_insights": ["No Claude client available - using fallback assessment"],
                    "improvement_areas": ["General prompt improvement"],
                    "risk_assessment": "Unknown without AI analysis",
                    "recommendation": "Configure Claude client for detailed analysis"
                }
            
            system_prompt = """You are an expert prompt optimization analyst. Analyze the provided context and assess optimization potential.

Respond with JSON in this format:
{
    "potential_score": 0.85,
    "key_insights": ["List of key insights about the prompt and performance"],
    "improvement_areas": ["Specific areas that could be improved"],
    "risk_assessment": "Assessment of optimization risks",
    "recommendation": "High-level recommendation for optimization approach"
}

Focus on actionable insights that would help guide optimization strategy."""

            context_message = self._build_context_analysis_message(context, None)
            
            response = await self.claude_client.analyze_context(
                system_prompt=system_prompt,
                context_data=context_message,
                component="orchestrator_potential_analysis"
            )
            
            content = response.get("content", "{}")
            if isinstance(content, str):
                return json.loads(content)
            return content
            
        except Exception as e:
            print(f"⚠️ Potential analysis failed: {str(e)}")
            return {
                "potential_score": 0.5,
                "key_insights": ["Analysis failed - using fallback assessment"],
                "improvement_areas": ["General prompt improvement"],
                "risk_assessment": "Unknown due to analysis failure",
                "recommendation": "Proceed with caution using default optimization"
            }
    
    async def close(self):
        """Close any resources"""
        if self.claude_client:
            await self.claude_client.close() 