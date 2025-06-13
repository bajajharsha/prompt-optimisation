"""
Optimization Controller - Combines existing optimization components
"""

import sys
import os
from typing import Dict, List, Any, Optional
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.core.context_manager import ContextManager
from prompt_optimizer.core.orchestrator import Orchestrator
from prompt_optimizer.core.simple_executor import SimpleExecutor
from prompt_optimizer.models.types import ModelConfiguration, OptimizationContext
from prompt_optimizer.utils.claude_client import ClaudeClient
from poc.intent_analysis.claude_intent_identifier import ClaudeIntentIdentifier


class OptimizationController:
    """
    Controls the optimization process using existing components
    """
    
    def __init__(self):
        self.claude_client = ClaudeClient()
        self.context_manager = ContextManager(self.claude_client)
        self.orchestrator = Orchestrator(self.claude_client)
        self.executor = SimpleExecutor(self.claude_client)
        self.intent_identifier = ClaudeIntentIdentifier(schema={}, baseline_metrics={})
    
    async def analyze_intent(
        self,
        schema: Dict[str, List[str]],
        baseline_metrics: Dict[str, Any],
        train_samples: List[Dict[str, Any]],
        base_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze intent using existing Claude intent identifier
        
        Args:
            schema: JSON schema
            baseline_metrics: Baseline metrics
            train_samples: Training samples for context
            base_prompt: Base prompt
            
        Returns:
            Intent analysis results
        """
        print("🎯 Running intent analysis...")
        
        # Update intent identifier with current data
        self.intent_identifier.schema = schema
        self.intent_identifier.baseline_metrics = baseline_metrics
        
        # Prepare analysis context
        analysis_context = {
            'schema': schema,
            'baseline_metrics': baseline_metrics,
            'failed_cases': baseline_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', [])[:3],
            'failed_cases_summary': baseline_metrics.get('failed_cases_summary', {}),
            'train_samples': train_samples,
            'base_prompt': base_prompt
        }
        
        # Run intent analysis
        intent_analysis = await self.intent_identifier.analyze_intent(analysis_context)
        
        # Convert to dict format
        intent_dict = {
            "classification_intent": intent_analysis.classification_intent,
            "schema_understanding": intent_analysis.schema_understanding,
            "domain_insights": intent_analysis.domain_insights,
            "classification_challenges": intent_analysis.classification_challenges
        }
        
        print("✅ Intent analysis completed")
        return intent_dict
    
    async def generate_candidates(
        self,
        current_prompt: str,
        current_metrics: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        schema: Dict[str, List[str]],
        iteration: int
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate prompts using existing optimization pipeline
        
        Args:
            current_prompt: Current prompt
            current_metrics: Current metrics
            intent_analysis: Intent analysis results
            schema: JSON schema
            iteration: Current iteration number
            
        Returns:
            List of candidate prompts with metadata
        """
        print(f"📝 Generating candidates for iteration {iteration}...")
        
        # Create optimization context
        context = self._create_optimization_context(
            current_prompt=current_prompt,
            current_metrics=current_metrics,
            intent_analysis=intent_analysis,
            schema=schema,
            iteration=iteration
        )
        
        # Select optimization strategy
        try:
            strategy_selection = await self.orchestrator.select_optimization_strategy(context)
            selected_optimizers = strategy_selection.selected_optimizers
            print(f"✅ Strategy selected: {', '.join(selected_optimizers)}")
        except Exception as e:
            print(f"⚠️  Strategy selection failed: {e}, using fallback")
            selected_optimizers = ["freeform_optimizer"]
        
        # Execute optimizers
        execution_results = await self.executor.execute_optimizers(
            optimizer_names=selected_optimizers,
            context=context
        )
        
        # Convert results to candidate format
        candidates = []
        for result in execution_results.results:
            if result.status.value == "completed":
                candidate = {
                    "strategy": result.optimizer_name,
                    "optimized_prompt": result.candidate_prompt,
                    "reasoning": result.reasoning,
                    "confidence": result.confidence,
                    "changes_made": result.changes_made,
                    "execution_time": result.execution_time
                }
                candidates.append(candidate)
        
        print(f"✅ Generated {len(candidates)} candidate prompts")
        return candidates
    
    def _create_optimization_context(
        self,
        current_prompt: str,
        current_metrics: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        schema: Dict[str, List[str]],
        iteration: int
    ) -> OptimizationContext:
        """
        Create optimization context for the current iteration
        
        Args:
            current_prompt: Current prompt
            current_metrics: Current metrics
            intent_analysis: Intent analysis
            schema: JSON schema
            iteration: Iteration number
            
        Returns:
            OptimizationContext object
        """
        # Extract failed cases
        failed_cases = current_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', [])
        failed_cases_summary = current_metrics.get('failed_cases_summary', {})
        
        current_metrics = {k: v for k, v in current_metrics.items() if k != 'detailed_failed_cases'}
        
        # Create target model configuration
        target_model = ModelConfiguration(
            provider="groq",
            model_name="llama-3.3-70b-versatile"
        )
        
        # Create context using context manager
        context = self.context_manager.create_initial_context(
            json_schema=schema,
            failed_cases_summary=failed_cases_summary,
            failed_cases=failed_cases,
            baseline_metrics=current_metrics,
            intent=intent_analysis,
            base_prompt=current_prompt,
            target_model=target_model
        )
        
        # Update iteration number
        context.iteration_number = iteration
        
        return context
    
    async def close(self):
        """Close all resources"""
        if self.claude_client:
            await self.claude_client.close()
        if self.context_manager:
            await self.context_manager.close()
        if self.orchestrator:
            await self.orchestrator.close()
        if self.executor:
            await self.executor.close() 