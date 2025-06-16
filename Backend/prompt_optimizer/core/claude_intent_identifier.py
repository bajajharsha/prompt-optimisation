"""
Claude-powered Intent Identification and Schema Usage Analysis for Prompt Optimization

This script uses Anthropic Claude API to intelligently analyze schema usage patterns 
and provides targeted feedback for prompt optimization strategies.
"""

import json
import os
import random
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import asyncio
from dataclasses import dataclass
from datetime import datetime
import httpx
from dotenv import load_dotenv
from pymongo import MongoClient
import pytz
from prompt_optimizer.core.request_id import get_request_id

load_dotenv()

@dataclass
class IntentAnalysis:
    """Data class for intent analysis results"""
    classification_intent: str
    schema_understanding: str
    domain_insights: str
    classification_challenges: str

class ClaudeIntentIdentifier:
    """
    Uses Anthropic Claude API to identify the intent and usage patterns 
    of classification schema for targeted optimization recommendations.
    """
    
    def __init__(self, schema: Dict[str, List[str]], baseline_metrics: Dict[str, Any]):
        """
        Initialize intent identifier with schema and baseline metrics.
        
        Args:
            schema: Classification schema with field names and enum values
            baseline_metrics: Baseline evaluation metrics from evaluation
        """
        self.schema = schema
        self.baseline_metrics = baseline_metrics
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    async def analyze_intent(self, analysis_context: Dict[str, Any]) -> IntentAnalysis:
        """
        Use Claude to perform comprehensive intent analysis.
        
        Args:
            failed_cases: Dictionary of failed cases by failure type
            train_samples: Optional sample training data for pattern analysis
            base_prompt: Optional base prompt for context analysis
            
        Returns:
            IntentAnalysis object with Claude's insights
        """
        print("🔍 Starting Claude-powered Intent Identification Analysis...")
        
        # Get Claude's analysis
        claude_response = await self._get_claude_analysis(analysis_context)
        
        # Parse Claude's response into structured format
        analysis = self._parse_claude_response(claude_response)
        
        return analysis
    
    async def _get_claude_analysis(self, context: Dict[str, Any]) -> str:
        """Send analysis request to Claude using httpx."""
        
        prompt = self._build_analysis_prompt(context)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 14000,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }
        
        from datetime import datetime
        client = MongoClient("mongodb://localhost:27017/")
        db = client["personal_project_log_usage"]
        collection = db["llm_usage"]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                
                result = response.json()
                
                log_entry = {
                    "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "input_tokens": result.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": result.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": result.get("usage", {}).get("input_tokens", 0) + result.get("usage", {}).get("output_tokens", 0),
                    "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/prompt_optimizer/core/claude_intent_identifier.py"
                }
                collection.insert_one(log_entry)
                
                print("Intent Identification Analysis:")
                print("="*60)
                print(result["content"][0]["text"])
                print("="*60)
                return result["content"][0]["text"]
                
        except httpx.HTTPStatusError as e:
            print(f"HTTP error calling Claude API: {e}")
            print(f"Response: {e.response.text}")
            raise
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            raise
    
    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive analysis prompt for Claude based on your optimization approach."""
        
        # Extract data from the analysis context
        schema = context.get('schema', {})
        baseline_metrics = context.get('baseline_metrics', {})
        failed_cases = context.get('failed_cases', [])
        failed_cases_summary = context.get('failed_cases_summary', {})
        train_samples = context.get('train_samples', [])
        base_prompt = context.get('base_prompt', '')
        
                # Build intent identification prompt
        prompt = f"""You are an expert classification analyst. Your task is to analyze a JSON classification system and understand its primary intent, domain, and how the classification should work.

## CLASSIFICATION SCHEMA
{json.dumps(schema, indent=2)}

## CURRENT BASE PROMPT
{base_prompt}

## TRAINING SAMPLES (5 Random Examples from LangFuse)
{json.dumps(train_samples, indent=2) if train_samples else "No training samples provided"}

## PERFORMANCE BASELINE
Current Performance: {json.dumps(baseline_metrics.get('summary', {}), indent=2)}
Field Performance: {json.dumps(baseline_metrics.get('enum_field_metrics', {}), indent=2)}

## FAILED CASES ANALYSIS
Failed Cases Summary: {json.dumps(failed_cases_summary, indent=2)}

Sample Failed Cases:
{json.dumps(failed_cases[:3], indent=2) if failed_cases else "No failed cases provided"}

## INTENT IDENTIFICATION TASK
Analyze the above information to understand:
1. What is this classification system trying to achieve?
2. What domain/industry does it serve?
3. How should the classification logic work?
4. What are the key decision criteria for each field?

Provide your analysis in the following JSON format:

```json
{{
  "classification_intent": {{
    "primary_purpose": "What is the main goal of this classification system?",
  }},
  "schema_understanding": {{
    "field_purposes": {{
      "field_name": "What this field is meant to classify and why it matters"
    }},
    "classification_logic": "How should users decide between the enum values?",
    "decision_criteria": "What information should guide the classification decisions?",
    "field_relationships": "How do the different fields relate to each other?"
  }},
  "domain_insights": {{
    "input_patterns": "What patterns do you see in the training data inputs?",
  }},
  "classification_challenges": {{
    "ambiguous_cases": "What makes classification difficult in this domain?",
    "edge_cases": "What edge cases should be considered?",
    "human_decision_factors": "What would a human expert consider when classifying?",
    "current_confusion_points": "Based on failed cases, where is the system getting confused?"
  }}
}}
```
## ANALYSIS GUIDELINES
1. **Focus on UNDERSTANDING the task**: What is this system supposed to do?
2. **Domain expertise**: Think like a domain expert who understands this classification problem
3. **Human perspective**: How would a human approach this classification task?
4. **Pattern recognition**: Identify consistent patterns in training data and failures

Your goal is to deeply understand the classification intent so that optimization strategies can be properly targeted."""

        return prompt
    
    def _parse_claude_response(self, claude_response: str) -> IntentAnalysis:
        """Parse Claude's response into structured IntentAnalysis object."""
        
        try:
            # Extract JSON from Claude's response
            json_start = claude_response.find('```json')
            json_end = claude_response.find('```', json_start + 7)
            
            if json_start != -1 and json_end != -1:
                json_str = claude_response[json_start + 7:json_end].strip()
            else:
                # Fallback: try to find JSON object directly
                json_start = claude_response.find('{')
                json_end = claude_response.rfind('}') + 1
                json_str = claude_response[json_start:json_end]
            
            claude_analysis = json.loads(json_str)
            # Save to request-specific intermediate results folder
            request_id = get_request_id() or "default"
            claude_analysis_dir = f"fastapi_optimization_system/intermediate_results/{request_id}/claude_analysis"
            os.makedirs(claude_analysis_dir, exist_ok=True)
            with open(f"{claude_analysis_dir}/claude_analysis.json", "w") as f:
                json.dump(claude_analysis, f, indent=2)
            # Create IntentAnalysis object
            analysis = IntentAnalysis(
                classification_intent=claude_analysis.get("classification_intent", "unknown"),
                schema_understanding=claude_analysis.get("schema_understanding", "unknown"),
                domain_insights=claude_analysis.get("domain_insights", "unknown"),
                classification_challenges=claude_analysis.get("classification_challenges", "unknown"),
            )
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Claude response as JSON: {e}")
            print(f"Claude response: {claude_response}")
            
            # Fallback: create basic analysis
            return IntentAnalysis(
                primary_intent="unknown",
                confidence_score=0.0,
                schema_complexity="moderate",
                critical_fields=[],
                failure_patterns={"error": "Failed to parse Claude response"},
                optimization_priorities=["ERROR: Could not analyze with Claude"],
                context_insights={"error": "JSON parsing failed"},
                claude_analysis={"raw_response": claude_response, "error": str(e)}
            )

    
    def _extract_high_priority_failures(self, analysis: IntentAnalysis) -> List[str]:
        """Extract high priority failures from Claude's analysis."""
        priorities = analysis.optimization_priorities
        high_priority = []
        
        for priority in priorities:
            if priority.startswith("URGENT:") or priority.startswith("HIGH:"):
                high_priority.append(priority)
        
        return high_priority
    
    def _determine_urgency(self, analysis: IntentAnalysis) -> str:
        """Determine optimization urgency based on Claude's analysis."""
        priorities = analysis.optimization_priorities
        
        urgent_count = sum(1 for p in priorities if p.startswith("URGENT:"))
        high_count = sum(1 for p in priorities if p.startswith("HIGH:"))
        
        if urgent_count > 0:
            return "urgent"
        elif high_count >= 2:
            return "high"
        elif high_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total possible combinations in schema."""
        total = 1
        for field, enum_values in self.schema.items():
            total *= len(enum_values)
        return total

    def save_analysis(self, analysis: IntentAnalysis, filepath: str = "claude_intent_analysis.json"):
        """Save the Claude intent analysis to a file."""
        analysis_dict = {
            "classification_intent": analysis.classification_intent,
            "schema_understanding": analysis.schema_understanding,
            "domain_insights": analysis.domain_insights,
            "classification_challenges": analysis.classification_challenges,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Claude intent analysis saved to: {filepath}")
        return filepath

# Utility functions for loading data
def load_baseline_metrics(filepath: str) -> Dict[str, Any]:
    """Load baseline metrics from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Baseline metrics file not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"⚠️  Error reading baseline metrics file: {filepath}")
        return {}

def fetch_train_samples_from_langfuse(dataset_name: str, limit: int = 5) -> List[Dict]:
    """
    Fetch train samples from LangFuse dataset.
    
    Args:
        dataset_name: Name of the dataset in LangFuse
        limit: Number of random samples to fetch
        
    Returns:
        List of train samples with input and expected_output
    """
    from langfuse import Langfuse
    import random
    
    langfuse_client = Langfuse(
        secret_key="sk-lf-d87cc28d-5a97-4fd9-bccd-13cfbf5e6ad3",
        public_key="pk-lf-4e626ffa-7bcd-495b-9f4d-f2f2c5b15087",
        host="https://cloud.langfuse.com"
    )
    
    try:
        print(f"📥 Fetching train samples from LangFuse dataset: {dataset_name}")
        dataset = langfuse_client.get_dataset(dataset_name)
        
        # Collect all items first
        all_items = []
        for item in dataset.items:
            try:
                # Parse expected output
                if isinstance(item.expected_output, dict):
                    expected_output = item.expected_output
                elif isinstance(item.expected_output, str):
                    expected_output = json.loads(item.expected_output)
                else:
                    continue
                
                train_sample = {
                    "input": item.input,
                    "expected_output": expected_output
                }
                all_items.append(train_sample)
                
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f"⚠️  Error parsing LangFuse item: {e}")
                continue
        
        # Randomly sample the requested number
        if len(all_items) <= limit:
            selected_samples = all_items
        else:
            selected_samples = random.sample(all_items, limit)
        
        print(f"✅ Successfully fetched {len(selected_samples)} train samples")
        return selected_samples
        
    except Exception as e:
        print(f"❌ Error fetching train samples from LangFuse: {e}")
        return []

# Example usage
async def main():
    """Example usage of Claude Intent Identifier."""
    
    # Example schema (replace with your actual schema)
    EXAMPLE_SCHEMA = {
        "action": ["CODE_GENERATION", "NOT_FOUND"],
        "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
        "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", 
                    "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
        "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
        "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
    }
    
    # Load data files
    # Load from request-specific intermediate results folder
    request_id = get_request_id() or "default"
    all_metrics = load_baseline_metrics(f"fastapi_optimization_system/intermediate_results/{request_id}/enhanced_baseline_results.json")
    
    failed_cases_summary = all_metrics.get("failed_cases_summary", {})
    failed_cases = all_metrics["detailed_failed_cases"]["wrong_classifications"]
    failed_cases = random.sample(failed_cases, 3)
    print("="*60)
    print(failed_cases)
    print("="*60)
    
    # baseline_metrics will everything from all_metrics other than detailed_failed_cases
    baseline_metrics = {k: v for k, v in all_metrics.items() if k != "detailed_failed_cases"}

    
    # Fetch train samples from LangFuse
    train_samples = fetch_train_samples_from_langfuse("code_gen", limit=3)
    print("="*60)
    print(f"Fetched {len(train_samples)} train samples from LangFuse:")
    print(train_samples)
    print("="*60)  

    # Example base prompt
    base_prompt = """
    You are an expert code generation classifier. Analyze the user's request and classify it according to the provided schema.
    Return your response as a valid JSON object with the specified fields.
    
    Create a page for jpeg to png image converter.
    
    IMPORTANT: Respond with a valid JSON object only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON.
    """
    
    analysis_context = {
        "failed_cases": failed_cases,
        "failed_cases_summary": failed_cases_summary,
        "train_samples": train_samples,
        "baseline_metrics": baseline_metrics,
        "base_prompt": base_prompt,
        "schema": EXAMPLE_SCHEMA
    }  
    # Run Claude intent identification
    identifier = ClaudeIntentIdentifier(EXAMPLE_SCHEMA, baseline_metrics)
    analysis = await identifier.analyze_intent(analysis_context)
    
    # Save results
    # Save to request-specific intermediate results folder
    request_id = get_request_id() or "default"
    filepath = identifier.save_analysis(analysis, f"fastapi_optimization_system/intermediate_results/{request_id}/claude_intent_analysis_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 CLAUDE INTENT IDENTIFICATION SUMMARY")
    print(filepath)
    print("="*60)
    
    return analysis

if __name__ == "__main__":
    asyncio.run(main())