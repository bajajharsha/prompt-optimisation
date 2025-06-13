#!/usr/bin/env python3
"""
Test script for enhanced JSON generation evaluation metrics.
Demonstrates the new optimization-focused features.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metric import JSONGenerationEvaluator, fetch_data_from_langfuse, run_groq_inference, extract_input_prompts
from langfuse import Langfuse


def test_enhanced_evaluation():
    """Test the enhanced evaluation with detailed failed case tracking."""
    
    # Static schema definition
    schema = {
        "action": ["CODE_GENERATION", "NOT_FOUND"],
        "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
        "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", 
                    "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
        "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
        "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
    }
    
    # Initialize enhanced evaluator
    evaluator = JSONGenerationEvaluator(schema)
    
    # Sample data for testing
    ground_truth = [
        {"action": "CODE_GENERATION", "subAction": "CODING", "platform": "DYNAMIC_WEB_APPLICATION", 
         "framework": "REACT", "languageType": "REACT_JAVASCRIPT"},
        {"action": "NOT_FOUND", "subAction": "GENERAL", "platform": "NOT_FOUND", 
         "framework": "NOT_FOUND", "languageType": "NOT_FOUND"},
        {"action": "CODE_GENERATION", "subAction": "VISUAL_EDITS", "platform": "STATIC_WEB_APPLICATION", 
         "framework": "NOT_FOUND", "languageType": "NOT_FOUND"}
    ]
    
    # Sample predictions with various types of errors
    predictions = [
        '{"action": "CODE_GENERATION", "subAction": "CODING", "platform": "WEB_APP", "framework": "REACT"}',  # Missing field + invalid enum
        'invalid json format',  # Invalid JSON
        '{"action": "CODE_GENERATION", "subAction": "VISUAL_EDITS", "platform": "STATIC_WEB_APPLICATION", "framework": "NOT_FOUND", "languageType": "NOT_FOUND"}'  # Correct
    ]
    
    input_prompts = [
        "Create a React web application for user management",
        "Hello, how are you?",
        "Design a login form UI"
    ]
    
    # Run enhanced evaluation
    results = evaluator.evaluate_batch(ground_truth, predictions, input_prompts)
    
    # Print detailed report
    evaluator.print_detailed_report(results)
    
    # Generate optimization report
    print("\n" + "="*80)
    print("OPTIMIZATION ANALYSIS REPORT")
    print("="*80)
    
    optimization_report = evaluator.generate_optimization_report(results)
    print(optimization_report)
    
    # Save failed cases for analysis
    evaluator.save_failed_cases("failed_cases_analysis.json", format="json")
    print(f"\n💾 Failed cases saved to: failed_cases_analysis.json")
    
    # Show field insights
    print(f"\n🔍 FIELD INSIGHTS SUMMARY:")
    for field, insights in results["field_insights"].items():
        print(f"  {field}: {insights['error_severity']} severity, {insights['missing_rate']*100:.1f}% missing rate")
    
    # Show optimization opportunities
    print(f"\n💡 OPTIMIZATION OPPORTUNITIES:")
    for category, opportunities in results["optimization_opportunities"].items():
        if opportunities:
            print(f"  {category.replace('_', ' ').title()}: {len(opportunities)} suggestions")


def test_baseline_with_langfuse():
    """Test with actual LangFuse data using enhanced metrics."""
    
    schema = {
        "action": ["CODE_GENERATION", "NOT_FOUND"],
        "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
        "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", 
                    "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
        "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
        "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
    }
    
    # schema = {
    #     "sentiment": ["POSITIVE", "NEGATIVE"],
    # }
    
    # Base prompt
    # base_prompt = """
    # You are a classification model. Classify the input into the correct category. Return the result in JSON format. 
    # """
    base_prompt = """
    You are a code generation request classifier. Your job is to analyze user requests and determine what type of code generation is needed.

Classify each request using this hierarchical approach:

1. **ACTION**: Is this actually requesting code generation?
   - CODE_GENERATION: User wants code to be created/generated
   - NOT_FOUND: Not a code generation request

2. **SUB-ACTION**: What type of coding work is needed?
   - CODING: Functional programming, logic, backend work
   - VISUAL_EDITS: UI/design focused work, styling, layouts
   - ERROR: Debugging, fixing errors, troubleshooting
   - GENERAL: General programming questions or non-specific tasks

3. **PLATFORM**: What is the target deployment platform?
   - DYNAMIC_WEB_APPLICATION: Web apps with interactive features, databases, user accounts
   - STATIC_WEB_APPLICATION: Simple web pages, portfolios, landing pages
   - DYNAMIC_MOBILE_APP: Mobile apps with interactive features, data storage
   - STATIC_MOBILE_APP: Simple mobile apps, basic functionality
   - NOT_FOUND: Platform not specified or unclear

4. **FRAMEWORK**: Which specific technology is mentioned?
   - REACT: Only if "React" is explicitly mentioned
   - FLUTTER: Only if "Flutter" is explicitly mentioned  
   - NOT_FOUND: No framework specified or framework not supported

5. **LANGUAGE_TYPE**: What specific language combination?
   - REACT_JAVASCRIPT: Only if React and JavaScript are both clearly indicated
   - NOT_FOUND: Language not specified or not supported

**CRITICAL RULES:**
- Only assign specific values (REACT, FLUTTER, etc.) when explicitly mentioned in the request
- When in doubt or when details are missing, use NOT_FOUND
- Don't assume or infer frameworks - require explicit mention
- Generic requests like "create a web app" without framework specification should use NOT_FOUND for framework

Return only a valid JSON object with all five fields:

{
  "action": "...",
  "subAction": "...",
  "platform": "...",
  "framework": "...",
  "languageType": "..."
}
    """
    # base_prompt = """
    # You are given movie reviews and you need to classify the sentiment of the review. Return the result in JSON format.
    # The schema is as follows:
    # {
    #   "sentiment": ["POSITIVE", "NEGATIVE"]
    # }
    # """
    
    # Initialize LangFuse
    langfuse_client = Langfuse(
        secret_key="sk-lf-d87cc28d-5a97-4fd9-bccd-13cfbf5e6ad3",
        public_key="pk-lf-4e626ffa-7bcd-495b-9f4d-f2f2c5b15087",
        host="https://cloud.langfuse.com"
    )
    
    try:
        # Fetch dataset
        dataset_name = "code_gen"
        # dataset_name = "movie_reviews"
        dataset = langfuse_client.get_dataset(dataset_name)
        
        # Extract data
        input_prompts = []
        ground_truth_jsons = []
        
        for i, item in enumerate(dataset.items):
            if i >= 15:
                break
                
            input_prompts.append(item.input)
            
            if isinstance(item.expected_output, dict):
                ground_truth_jsons.append(item.expected_output)
            else:
                ground_truth_jsons.append(json.loads(item.expected_output))
        
        # Run inference
        predicted_texts = run_groq_inference(input_prompts, base_prompt)
        
        # Enhanced evaluation
        evaluator = JSONGenerationEvaluator(schema)
        results = evaluator.evaluate_batch(ground_truth_jsons, predicted_texts, input_prompts)
        
        # Print detailed analysis
        print("\n" + "="*80)
        print("ENHANCED BASELINE EVALUATION RESULTS")
        print("="*80)
        
        evaluator.print_detailed_report(results)
        
        # Generate and print optimization report
        optimization_report = evaluator.generate_optimization_report(results)
        print(f"\n{optimization_report}")
        
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs("metrics", exist_ok=True)
        
        # Save enhanced results
        with open("metrics/enhanced_baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save failed cases for optimization
        evaluator.save_failed_cases("metrics/baseline_failed_cases.json")
        
        print(f"\n💾 Enhanced results saved to: metrics/enhanced_baseline_results.json")
        print(f"💾 Failed cases saved to: metrics/baseline_failed_cases.json")
        
        return results
        
    except Exception as e:
        print(f"Error in LangFuse evaluation: {e}")
        return None


def demonstrate_comparison():
    """Demonstrate metric comparison between different prompt versions."""
    
    # Load baseline results (if available)
    baseline_file = "baseline_evaluation_results.json"
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        # Run enhanced evaluation to get new results
        new_results = test_baseline_with_langfuse()
        
        if new_results:
            # Compare metrics
            comparison = JSONGenerationEvaluator.compare_metrics(baseline_results, new_results)
            
            print(f"\n" + "="*80)
            print("BASELINE vs ENHANCED METRICS COMPARISON")
            print("="*80)
            
            overall = comparison["overall_improvement"]
            print(f"\n📊 OVERALL IMPROVEMENTS:")
            print(f"  Valid JSON:     {overall['valid_json_improvement']:+.3f}")
            print(f"  Exact Match:    {overall['exact_match_improvement']:+.3f}")
            print(f"  Average F1:     {overall['avg_f1_improvement']:+.3f}")
            
            if comparison["significant_improvements"]:
                print(f"\n🚀 SIGNIFICANT IMPROVEMENTS:")
                for improvement in comparison["significant_improvements"]:
                    print(f"  {improvement['field']}: {improvement['improvement']:+.3f} ({improvement['improvement_percentage']:+.1f}%)")
            
            if comparison["regression_alerts"]:
                print(f"\n⚠️  REGRESSION ALERTS:")
                for regression in comparison["regression_alerts"]:
                    print(f"  {regression['field']}: {regression['regression']:+.3f} ({regression['regression_percentage']:+.1f}%)")
    else:
        print(f"Baseline results file not found: {baseline_file}")


if __name__ == "__main__":
    print("Baseline Performance Evaluation")
    print("="*60)
    
    # Baseline evaluation with LangFuse Dateset
    print("\n\nTesting enhanced baseline evaluation with LangFuse Dataset...")
    test_baseline_with_langfuse()
    
    # Demonstrate comparison capabilities
    # print("\n\nDemonstrating metric comparison...")
    # demonstrate_comparison()
    
    print(f"\n All tests completed!") 