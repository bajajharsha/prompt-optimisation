from typing import Dict, List, Any, Tuple
import json
from trials.optimization_strategies import get_available_strategies, get_strategy_descriptions

class MetaAgent:
    """Meta-agent that selects and applies optimization strategies"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.strategies = get_available_strategies()
        self.strategy_descriptions = get_strategy_descriptions()
        self.optimization_history = []
    
    def analyze_and_optimize(self, 
                           base_prompt: str, 
                           failed_cases: List[Dict], 
                           schema: Dict, 
                           baseline_metrics: Dict,
                           context: Dict = None) -> Tuple[str, List[str]]:
        """
        Analyze failure patterns and generate optimized prompt
        
        Returns:
            Tuple of (optimized_prompt, selected_strategies)
        """
        if context is None:
            context = {}
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failures(failed_cases, baseline_metrics)
        
        # Select optimization strategies
        selected_strategies = self._select_strategies(failure_analysis, context)
        
        # Apply strategies in order
        optimized_prompt = self._apply_strategies(
            base_prompt, selected_strategies, failed_cases, schema, context
        )
        
        # Record optimization attempt
        self._record_optimization(base_prompt, optimized_prompt, selected_strategies, failure_analysis)
        
        return optimized_prompt, selected_strategies
    
    def _analyze_failures(self, failed_cases: List[Dict], baseline_metrics: Dict) -> Dict:
        """Analyze failure patterns to inform strategy selection"""
        analysis = {
            'total_failures': len(failed_cases),
            'failure_types': {},
            'severity_scores': {},
            'patterns': {}
        }
        
        # Count failure types
        for case in failed_cases:
            failure_type = case.get('failure_type', 'unknown')
            analysis['failure_types'][failure_type] = analysis['failure_types'].get(failure_type, 0) + 1
        
        # Calculate severity scores
        total_cases = baseline_metrics.get('total_cases', len(failed_cases))
        if total_cases > 0:
            analysis['severity_scores'] = {
                'json_validity_rate': 1 - (analysis['failure_types'].get('invalid_json', 0) / total_cases),
                'enum_compliance_rate': 1 - (analysis['failure_types'].get('enum_violation', 0) / total_cases),
                'classification_accuracy': 1 - (analysis['failure_types'].get('wrong_classification', 0) / total_cases)
            }
        
        # Identify specific patterns
        analysis['patterns'] = self._identify_patterns(failed_cases)
        
        return analysis
    
    def _identify_patterns(self, failed_cases: List[Dict]) -> Dict:
        """Identify specific failure patterns"""
        patterns = {
            'needs_examples': False,
            'needs_reasoning': False,
            'needs_constraints': False,
            'needs_role': False,
            'needs_error_refinement': False
        }
        
        # Check if examples would help
        classification_errors = sum(1 for case in failed_cases if case.get('failure_type') == 'wrong_classification')
        if classification_errors > len(failed_cases) * 0.3:  # >30% classification errors
            patterns['needs_examples'] = True
        
        # Check if reasoning would help
        complex_failures = sum(1 for case in failed_cases 
                             if len(case.get('input', '').split()) > 20)  # Long inputs
        if complex_failures > len(failed_cases) * 0.2:  # >20% complex cases
            patterns['needs_reasoning'] = True
        
        # Check if constraint reinforcement needed
        constraint_violations = sum(1 for case in failed_cases 
                                  if case.get('failure_type') in ['invalid_json', 'enum_violation'])
        if constraint_violations > len(failed_cases) * 0.2:  # >20% constraint violations
            patterns['needs_constraints'] = True
        
        # Check if role assignment would help
        if len(failed_cases) > 10 and classification_errors > 5:
            patterns['needs_role'] = True
        
        # Check if error pattern refinement needed
        if len(failed_cases) > 5:
            patterns['needs_error_refinement'] = True
        
        return patterns
    
    def _select_strategies(self, failure_analysis: Dict, context: Dict) -> List[str]:
        """Select optimization strategies based on failure analysis"""
        selected = []
        patterns = failure_analysis.get('patterns', {})
        failure_types = failure_analysis.get('failure_types', {})
        
        # Priority-based strategy selection
        
        # 1. Address constraint violations first (highest priority)
        if patterns.get('needs_constraints') or failure_types.get('invalid_json', 0) > 0 or failure_types.get('enum_violation', 0) > 0:
            selected.append('constraint_specification')
        
        # 2. Add role for better performance
        if patterns.get('needs_role'):
            selected.append('role_based')
        
        # 3. Add examples for classification improvement
        if patterns.get('needs_examples'):
            selected.append('few_shot_examples')
        
        # 4. Add reasoning for complex cases
        if patterns.get('needs_reasoning'):
            selected.append('chain_of_thought')
        
        # 5. Apply error pattern refinement for targeted fixes
        if patterns.get('needs_error_refinement'):
            selected.append('error_pattern_refinement')
        
        # Ensure at least one strategy is selected
        if not selected:
            # Default strategy based on most common failure type
            most_common_failure = max(failure_types.items(), key=lambda x: x[1])[0] if failure_types else 'unknown'
            if most_common_failure == 'wrong_classification':
                selected.append('few_shot_examples')
            elif most_common_failure in ['invalid_json', 'enum_violation']:
                selected.append('constraint_specification')
            else:
                selected.append('role_based')  # Safe default
        
        return selected
    
    def _apply_strategies(self, 
                         base_prompt: str, 
                         strategy_names: List[str], 
                         failed_cases: List[Dict], 
                         schema: Dict, 
                         context: Dict) -> str:
        """Apply selected strategies in sequence"""
        current_prompt = base_prompt
        
        for strategy_name in strategy_names:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                current_prompt = strategy.optimize(current_prompt, failed_cases, schema, context)
            else:
                print(f"Warning: Strategy '{strategy_name}' not found")
        
        return current_prompt
    
    def _record_optimization(self, 
                           base_prompt: str, 
                           optimized_prompt: str, 
                           strategies: List[str], 
                           failure_analysis: Dict):
        """Record optimization attempt for tracking"""
        record = {
            'timestamp': self._get_timestamp(),
            'strategies_used': strategies,
            'failure_analysis': failure_analysis,
            'prompt_length_change': len(optimized_prompt) - len(base_prompt),
            'base_prompt_hash': hash(base_prompt),
            'optimized_prompt_hash': hash(optimized_prompt)
        }
        
        self.optimization_history.append(record)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_optimization_history(self) -> List[Dict]:
        """Get history of optimization attempts"""
        return self.optimization_history
    
    def get_strategy_usage_stats(self) -> Dict:
        """Get statistics on strategy usage"""
        stats = {}
        for record in self.optimization_history:
            for strategy in record['strategies_used']:
                stats[strategy] = stats.get(strategy, 0) + 1
        return stats

# Advanced Meta-Agent with LLM-based strategy selection
class LLMMetaAgent(MetaAgent):
    """Meta-agent that uses LLM to make strategy selection decisions"""
    
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.llm_client = llm_client
    
    def _select_strategies(self, failure_analysis: Dict, context: Dict) -> List[str]:
        """Use LLM to select strategies based on failure analysis"""
        
        # Prepare context for LLM
        analysis_prompt = self._build_strategy_selection_prompt(failure_analysis, context)
        
        try:
            # Get LLM recommendation
            response = self.llm_client.generate(analysis_prompt)
            selected_strategies = self._parse_strategy_response(response)
            
            # Validate and fallback to rule-based if needed
            if not selected_strategies or not all(s in self.strategies for s in selected_strategies):
                print("LLM strategy selection failed, falling back to rule-based")
                return super()._select_strategies(failure_analysis, context)
            
            return selected_strategies
            
        except Exception as e:
            print(f"Error in LLM strategy selection: {e}, falling back to rule-based")
            return super()._select_strategies(failure_analysis, context)
    
    def _build_strategy_selection_prompt(self, failure_analysis: Dict, context: Dict) -> str:
        """Build prompt for LLM strategy selection"""
        prompt = """# Prompt Optimization Strategy Selection

You are an expert in prompt optimization for JSON classification tasks with enum constraints.

## Available Strategies:
"""
        
        for name, description in self.strategy_descriptions.items():
            prompt += f"- **{name}**: {description}\n"
        
        prompt += f"""

## Failure Analysis:
- Total failures: {failure_analysis.get('total_failures', 0)}
- Failure types: {json.dumps(failure_analysis.get('failure_types', {}), indent=2)}
- Severity scores: {json.dumps(failure_analysis.get('severity_scores', {}), indent=2)}

## Context:
{json.dumps(context, indent=2)}

## Task:
Select the most appropriate optimization strategies to address these failures. Consider:
1. Which failures are most critical to address
2. Which strategies would be most effective for the observed patterns
3. The order in which strategies should be applied
4. Avoid over-optimization (don't select too many strategies)

## Response Format:
Return ONLY a JSON array of strategy names in order of application:
["strategy1", "strategy2", ...]

Example: ["constraint_specification", "few_shot_examples"]
"""
        
        return prompt
    
    def _parse_strategy_response(self, response: str) -> List[str]:
        """Parse LLM response to extract strategy list"""
        try:
            # Try to extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                strategies = json.loads(json_match.group())
                return [s for s in strategies if isinstance(s, str) and s in self.strategies]
        except:
            pass
        
        # Fallback: look for strategy names in response
        found_strategies = []
        for strategy_name in self.strategies.keys():
            if strategy_name in response:
                found_strategies.append(strategy_name)
        
        return found_strategies 