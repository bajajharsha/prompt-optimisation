from abc import ABC, abstractmethod
from typing import Dict, List, Any
import json
import random

class OptimizationStrategy(ABC):
    """Base class for all prompt optimization strategies"""
    
    @abstractmethod
    def optimize(self, base_prompt: str, failed_cases: List[Dict], schema: Dict, context: Dict) -> str:
        """Apply optimization strategy to base prompt"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy identifier"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return strategy description for meta-agent"""
        pass

class StructuredOutputStrategy(OptimizationStrategy):
    """Optimize for structured JSON output with explicit formatting"""
    
    def optimize(self, base_prompt: str, failed_cases: List[Dict], schema: Dict, context: Dict) -> str:
        # Analyze JSON formatting failures
        json_failures = [case for case in failed_cases if case.get('failure_type') == 'invalid_json']
        
        # Build structured output section with explicit JSON template
        structured_section = self._build_structured_output_section(schema, json_failures)
        
        # Insert structured output instructions
        optimized_prompt = self._insert_structured_instructions(base_prompt, structured_section)
        
        return optimized_prompt
    
    def _build_structured_output_section(self, schema: Dict, json_failures: List[Dict]) -> str:
        """Build structured output instructions with JSON template"""
        section = "\n## STRUCTURED OUTPUT FORMAT:\n\n"
        
        # Add explicit JSON template
        enum_values = schema.get('enum_values', [])
        field_name = schema.get('field_name', 'classification')
        
        section += "### Required JSON Structure:\n"
        section += "```json\n"
        section += "{\n"
        section += f'  "{field_name}": "VALUE_FROM_ENUM_LIST"\n'
        section += "}\n"
        section += "```\n\n"
        
        # Add enum constraints
        if enum_values:
            section += f"### Valid Values for '{field_name}':\n"
            for i, value in enumerate(enum_values, 1):
                section += f"{i}. `\"{value}\"`\n"
            section += "\n"
        
        # Add JSON formatting rules based on failures
        if json_failures:
            section += "### JSON Formatting Rules:\n"
            section += "- Use EXACTLY double quotes (\") for strings, never single quotes\n"
            section += "- No trailing commas after the last element\n"
            section += "- Ensure proper bracket/brace matching: { }\n"
            section += "- No extra characters outside the JSON object\n"
            section += "- No comments or explanations in the JSON\n\n"
        
        # Add validation checklist
        section += "### Before Responding - Validate:\n"
        section += "1. Is it valid JSON? (use double quotes, proper brackets)\n"
        section += f"2. Is the '{field_name}' value exactly one of the allowed enum values?\n"
        section += "3. Are there any extra fields or missing required fields?\n"
        
        return section
    
    def _analyze_failure_patterns(self, failed_cases: List[Dict]) -> List[str]:
        """Identify common failure patterns"""
        patterns = []
        for case in failed_cases:
            if case.get('failure_type') == 'enum_violation':
                patterns.append(f"enum_violation_{case.get('attempted_value')}")
            elif case.get('failure_type') == 'invalid_json':
                patterns.append("invalid_json")
            elif case.get('failure_type') == 'wrong_classification':
                patterns.append(f"wrong_class_{case.get('expected')}_{case.get('predicted')}")
        return list(set(patterns))
    
    def _find_counter_examples(self, successful_examples: List[Dict], pattern: str) -> List[Dict]:
        """Find examples that counter specific failure patterns"""
        if pattern.startswith('enum_violation'):
            # Find examples with correct enum usage
            return [ex for ex in successful_examples if ex.get('output') in ex.get('valid_enums', [])]
        elif pattern == 'invalid_json':
            # Find examples with perfect JSON structure
            return [ex for ex in successful_examples if self._is_valid_json(ex.get('output_raw', ''))]
        return []
    
    def _is_valid_json(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except:
            return False
    
    def _build_few_shot_section(self, examples: List[Dict], schema: Dict) -> str:
        """Build the few-shot examples section"""
        section = "\n## Examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            section += f"Example {i}:\n"
            section += f"Input: {example.get('input', '')}\n"
            section += f"Output: {json.dumps(example.get('output', {}), indent=2)}\n\n"
        
        return section
    
    def _insert_structured_instructions(self, base_prompt: str, structured_section: str) -> str:
        """Insert structured output instructions into base prompt"""
        # Insert before any existing output format section or at the end
        if "## Output Format:" in base_prompt:
            return base_prompt.replace("## Output Format:", structured_section + "\n## Output Format:")
        elif "## Instructions:" in base_prompt:
            return base_prompt.replace("## Instructions:", f"{structured_section}\n## Instructions:")
        else:
            return base_prompt + "\n" + structured_section
    
    def get_strategy_name(self) -> str:
        return "structured_output"
    
    def get_description(self) -> str:
        return "Adds explicit JSON structure template and formatting rules to ensure valid JSON output with proper enum compliance"

class ClassificationReasoningStrategy(OptimizationStrategy):
    """Add step-by-step classification reasoning for better accuracy"""
    
    def optimize(self, base_prompt: str, failed_cases: List[Dict], schema: Dict, context: Dict) -> str:
        # Analyze classification errors and reasoning needs
        classification_errors = [case for case in failed_cases if case.get('failure_type') == 'wrong_classification']
        
        # Build classification reasoning section
        reasoning_section = self._build_classification_reasoning_section(schema, classification_errors, context)
        
        # Insert reasoning instructions
        optimized_prompt = self._insert_reasoning_instructions(base_prompt, reasoning_section)
        
        return optimized_prompt
    
    def _build_classification_reasoning_section(self, schema: Dict, classification_errors: List[Dict], context: Dict) -> str:
        """Build classification reasoning instructions"""
        section = "\n## CLASSIFICATION REASONING PROCESS:\n\n"
        
        enum_values = schema.get('enum_values', [])
        domain = context.get('domain', 'general')
        
        section += "Follow this systematic approach for accurate classification:\n\n"
        
        # Step 1: Text Analysis
        section += "### Step 1: Analyze the Input\n"
        section += "- Read the entire input text carefully\n"
        section += "- Identify key words, phrases, and context clues\n"
        section += "- Consider the overall tone and intent\n\n"
        
        # Step 2: Category Evaluation (domain-specific)
        section += "### Step 2: Evaluate Each Category\n"
        if enum_values:
            section += f"Consider each valid option systematically:\n"
            for value in enum_values:
                section += f"- **{value}**: Does the input match this category? Why or why not?\n"
            section += "\n"
        
        # Step 3: Elimination Process
        section += "### Step 3: Elimination Process\n"
        section += "- Rule out categories that clearly don't fit\n"
        section += "- Focus on the remaining candidates\n"
        section += "- Look for distinguishing features between similar categories\n\n"
        
        # Step 4: Final Decision
        section += "### Step 4: Make Final Decision\n"
        section += "- Select the BEST matching category from the valid options\n"
        section += "- Ensure your choice is one of the exact enum values\n"
        section += "- Double-check your reasoning\n\n"
        
        # Add domain-specific guidance
        if domain == 'intent_classification':
            section += "### Intent Classification Tips:\n"
            section += "- Focus on what the user wants to accomplish\n"
            section += "- Look for action words and request patterns\n"
            section += "- Consider the context and urgency\n\n"
        elif domain == 'sentiment_analysis':
            section += "### Sentiment Analysis Tips:\n"
            section += "- Look for emotional indicators and tone\n"
            section += "- Consider both explicit and implicit sentiment\n"
            section += "- Watch for sarcasm or mixed emotions\n\n"
        
        section += "### Step 5: Format Response\n"
        section += "- Format your final answer as valid JSON\n"
        section += "- Use the exact enum value (case-sensitive)\n"
        section += "- No additional text outside the JSON\n"
        
        return section
    
    def _insert_reasoning_instructions(self, base_prompt: str, reasoning_section: str) -> str:
        """Insert classification reasoning instructions into base prompt"""
        # Insert before output format or examples
        if "## STRUCTURED OUTPUT FORMAT:" in base_prompt:
            return base_prompt.replace("## STRUCTURED OUTPUT FORMAT:", f"{reasoning_section}\n## STRUCTURED OUTPUT FORMAT:")
        elif "## Output Format:" in base_prompt:
            return base_prompt.replace("## Output Format:", f"{reasoning_section}\n## Output Format:")
        elif "## Examples:" in base_prompt:
            return base_prompt.replace("## Examples:", f"{reasoning_section}\n## Examples:")
        else:
            return base_prompt + "\n" + reasoning_section
    
    def get_strategy_name(self) -> str:
        return "classification_reasoning"
    
    def get_description(self) -> str:
        return "Adds systematic step-by-step classification reasoning process with domain-specific guidance for better accuracy"

class RoleBasedStrategy(OptimizationStrategy):
    """Assign specific role/persona to improve performance"""
    
    def optimize(self, base_prompt: str, failed_cases: List[Dict], schema: Dict, context: Dict) -> str:
        # Determine best role based on domain and failures
        role = self._select_optimal_role(schema, failed_cases, context)
        
        # Build role section
        role_section = self._build_role_section(role, schema)
        
        # Insert at beginning of prompt
        optimized_prompt = role_section + "\n\n" + base_prompt
        
        return optimized_prompt
    
    def _select_optimal_role(self, schema: Dict, failed_cases: List[Dict], context: Dict) -> Dict:
        """Select the most appropriate role based on context"""
        domain = context.get('domain', 'general')
        
        roles = {
            'intent_classification': {
                'title': 'Expert Intent Classification Specialist',
                'description': 'You are an expert at understanding user intents and classifying them accurately.',
                'expertise': 'natural language understanding, intent recognition, and user behavior analysis'
            },
            'sentiment_analysis': {
                'title': 'Professional Sentiment Analyst',
                'description': 'You are a professional sentiment analyst with expertise in emotional tone detection.',
                'expertise': 'emotional intelligence, linguistic analysis, and sentiment classification'
            },
            'category_classification': {
                'title': 'Expert Content Categorization Specialist',
                'description': 'You are an expert at categorizing and organizing content into appropriate categories.',
                'expertise': 'content analysis, taxonomy design, and systematic categorization'
            },
            'general': {
                'title': 'Expert Classification Analyst',
                'description': 'You are an expert analyst specializing in accurate text classification.',
                'expertise': 'text analysis, pattern recognition, and systematic classification'
            }
        }
        
        return roles.get(domain, roles['general'])
    
    def _build_role_section(self, role: Dict, schema: Dict) -> str:
        """Build role assignment section"""
        section = f"# Role: {role['title']}\n\n"
        section += f"{role['description']} "
        section += f"Your expertise includes {role['expertise']}.\n\n"
        section += "Your task is to provide accurate classifications that strictly adhere to the given constraints and output format."
        
        return section
    
    def get_strategy_name(self) -> str:
        return "role_based"
    
    def get_description(self) -> str:
        return "Assigns an expert role/persona to improve classification performance and adherence to constraints"

class ConstraintSpecificationStrategy(OptimizationStrategy):
    """Strengthen constraint specification and validation"""
    
    def optimize(self, base_prompt: str, failed_cases: List[Dict], schema: Dict, context: Dict) -> str:
        # Analyze constraint violations
        violations = self._analyze_constraint_violations(failed_cases)
        
        # Build enhanced constraint section
        constraint_section = self._build_constraint_section(schema, violations)
        
        # Insert constraints prominently
        optimized_prompt = self._insert_constraints(base_prompt, constraint_section)
        
        return optimized_prompt
    
    def _analyze_constraint_violations(self, failed_cases: List[Dict]) -> Dict:
        """Analyze types of constraint violations"""
        violations = {
            'enum_violations': [],
            'json_format_errors': [],
            'missing_fields': [],
            'extra_fields': []
        }
        
        for case in failed_cases:
            failure_type = case.get('failure_type')
            if failure_type == 'enum_violation':
                violations['enum_violations'].append(case.get('attempted_value'))
            elif failure_type == 'invalid_json':
                violations['json_format_errors'].append(case.get('error_details'))
        
        return violations
    
    def _build_constraint_section(self, schema: Dict, violations: Dict) -> str:
        """Build enhanced constraint specification"""
        section = "\n## CRITICAL CONSTRAINTS:\n\n"
        
        # JSON format constraints
        section += "### JSON Format Requirements:\n"
        section += "- Your response MUST be valid JSON\n"
        section += "- Use double quotes for strings\n"
        section += "- No trailing commas\n"
        section += "- Proper escaping of special characters\n\n"
        
        # Enum constraints
        enum_values = schema.get('enum_values', [])
        if enum_values:
            section += "### Valid Values (ENUM CONSTRAINTS):\n"
            section += f"- You MUST only use these exact values: {enum_values}\n"
            section += "- Any other value will be considered INVALID\n"
            section += "- Values are case-sensitive\n"
            section += "- No variations or synonyms allowed\n\n"
        
        # Address specific violations
        if violations['enum_violations']:
            section += "### Common Mistakes to AVOID:\n"
            for violation in set(violations['enum_violations']):
                section += f"- DO NOT use '{violation}' - it's not in the valid set\n"
            section += "\n"
        
        # Validation reminder
        section += "### Validation:\n"
        section += "- Double-check your output against these constraints\n"
        section += "- Ensure JSON is valid and parseable\n"
        section += "- Verify your value is in the allowed enum set\n"
        
        return section
    
    def _insert_constraints(self, base_prompt: str, constraint_section: str) -> str:
        """Insert constraints prominently in prompt"""
        # Insert after role/intro but before examples
        if "## Examples:" in base_prompt:
            return base_prompt.replace("## Examples:", f"{constraint_section}\n## Examples:")
        elif "## Instructions:" in base_prompt:
            return base_prompt.replace("## Instructions:", f"{constraint_section}\n## Instructions:")
        else:
            return base_prompt + constraint_section
    
    def get_strategy_name(self) -> str:
        return "constraint_specification"
    
    def get_description(self) -> str:
        return "Strengthens constraint specification with explicit validation rules and common mistake prevention"

class ErrorPatternRefinementStrategy(OptimizationStrategy):
    """Refine prompt based on specific error patterns"""
    
    def optimize(self, base_prompt: str, failed_cases: List[Dict], schema: Dict, context: Dict) -> str:
        # Analyze error patterns in detail
        error_patterns = self._analyze_detailed_error_patterns(failed_cases)
        
        # Generate targeted refinements
        refinements = self._generate_targeted_refinements(error_patterns, schema)
        
        # Apply refinements to prompt
        optimized_prompt = self._apply_refinements(base_prompt, refinements)
        
        return optimized_prompt
    
    def _analyze_detailed_error_patterns(self, failed_cases: List[Dict]) -> Dict:
        """Perform detailed analysis of error patterns"""
        patterns = {
            'confusion_pairs': {},  # Which enums are confused with each other
            'input_patterns': {},   # What input patterns lead to errors
            'json_errors': [],      # Specific JSON formatting issues
            'semantic_errors': []   # Semantic misunderstandings
        }
        
        for case in failed_cases:
            failure_type = case.get('failure_type')
            
            if failure_type == 'wrong_classification':
                expected = case.get('expected')
                predicted = case.get('predicted')
                if expected and predicted:
                    key = f"{expected}->{predicted}"
                    patterns['confusion_pairs'][key] = patterns['confusion_pairs'].get(key, 0) + 1
            
            elif failure_type == 'invalid_json':
                patterns['json_errors'].append(case.get('error_details', ''))
            
            # Analyze input patterns
            input_text = case.get('input', '')
            if input_text:
                # Simple pattern detection (can be enhanced)
                if len(input_text.split()) < 5:
                    patterns['input_patterns']['short_text'] = patterns['input_patterns'].get('short_text', 0) + 1
                elif '?' in input_text:
                    patterns['input_patterns']['question'] = patterns['input_patterns'].get('question', 0) + 1
        
        return patterns
    
    def _generate_targeted_refinements(self, error_patterns: Dict, schema: Dict) -> List[str]:
        """Generate specific refinements based on error patterns"""
        refinements = []
        
        # Address confusion pairs
        confusion_pairs = error_patterns.get('confusion_pairs', {})
        if confusion_pairs:
            refinements.append("### Disambiguation Guidelines:")
            for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
                if count > 1:  # Only address frequent confusions
                    expected, predicted = pair.split('->')
                    refinements.append(f"- When choosing between '{expected}' and '{predicted}', carefully consider [specific guidance needed]")
        
        # Address input pattern issues
        input_patterns = error_patterns.get('input_patterns', {})
        if input_patterns.get('short_text', 0) > 2:
            refinements.append("- For short inputs, pay extra attention to context clues and implicit meaning")
        if input_patterns.get('question', 0) > 2:
            refinements.append("- For questions, focus on the intent behind the question rather than just the surface form")
        
        # Address JSON errors
        json_errors = error_patterns.get('json_errors', [])
        if json_errors:
            refinements.append("### JSON Formatting Reminders:")
            refinements.append("- Always use double quotes, never single quotes")
            refinements.append("- Ensure proper bracket/brace matching")
            refinements.append("- No trailing commas after the last element")
        
        return refinements
    
    def _apply_refinements(self, base_prompt: str, refinements: List[str]) -> str:
        """Apply refinements to the base prompt"""
        if not refinements:
            return base_prompt
        
        refinement_section = "\n## Error Pattern Refinements:\n\n"
        refinement_section += "\n".join(refinements)
        refinement_section += "\n"
        
        # Insert before output format
        if "## Output Format:" in base_prompt:
            return base_prompt.replace("## Output Format:", f"{refinement_section}\n## Output Format:")
        else:
            return base_prompt + refinement_section
    
    def get_strategy_name(self) -> str:
        return "error_pattern_refinement"
    
    def get_description(self) -> str:
        return "Applies targeted refinements based on detailed analysis of specific error patterns and failure modes"

# Strategy Registry
OPTIMIZATION_STRATEGIES = {
    'few_shot_examples': FewShotExampleStrategy(),
    'chain_of_thought': ChainOfThoughtStrategy(),
    'role_based': RoleBasedStrategy(),
    'constraint_specification': ConstraintSpecificationStrategy(),
    'error_pattern_refinement': ErrorPatternRefinementStrategy()
}

def get_available_strategies() -> Dict[str, OptimizationStrategy]:
    """Get all available optimization strategies"""
    return OPTIMIZATION_STRATEGIES

def get_strategy_descriptions() -> Dict[str, str]:
    """Get descriptions of all strategies for meta-agent"""
    return {name: strategy.get_description() for name, strategy in OPTIMIZATION_STRATEGIES.items()} 