#!/usr/bin/env python3
"""
Complete Prompt Optimization Pipeline
Runs the entire system end-to-end using your existing context setup
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from prompt_optimizer.core.context_manager import ContextManager
from prompt_optimizer.core.orchestrator import Orchestrator
from prompt_optimizer.core.simple_executor import SimpleExecutor
from prompt_optimizer.models.types import ModelConfiguration
from prompt_optimizer.utils.claude_client import ClaudeClient
from poc.intent_analysis.claude_intent_identifier import load_baseline_metrics


async def run_optimization_pipeline():
    """
    Complete optimization pipeline using your existing context pattern
    """
    
    print("🚀 Starting Prompt Optimization Pipeline")
    print("=" * 60)
    
    # Initialize components
    print("📦 Initializing components...")
    claude_client = ClaudeClient()
    context_manager = ContextManager(claude_client)
    orchestrator = Orchestrator(claude_client)
    executor = SimpleExecutor(claude_client)
    
    # Load your existing data (same as your pattern)
    print("📊 Loading baseline data...")
    all_metrics = load_baseline_metrics("poc/metrics/enhanced_baseline_results.json")
    
    with open("poc/intent_analysis/claude_intent_analysis_results.json", 'r', encoding='utf-8') as f:
        intent_analysis = json.load(f)
    
    # Your base prompt
    base_prompt = """
    You are a classification model. Classify the input into the correct category. Return the result in JSON format. 
    The schema is as follows:
    {{
    "action": ["CODE_GENERATION", "NOT_FOUND"],
    "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
    "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
    "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
    "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
    }}
    
    IMPORTANT: Respond with a valid JSON object only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON.
    """
    
    # Step 1: Create Initial Context (using your exact pattern)
    print("\n🔧 Step 1: Creating initial context...")
    
    context = context_manager.create_initial_context(
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
            model_name="llama-3.1-70b-versatile",
        )
    )
    
    # Step 2: Strategy Selection
    print("\n🎯 Step 2: Selecting optimization strategy...")
    try:
        strategy_selection = await orchestrator.select_optimization_strategy(context)
        
        print(f"✅ Strategy selected:")
        print(f"   Optimizers: {', '.join(strategy_selection.selected_optimizers)}")
        
    except Exception as e:
        print(f"⚠️ Strategy selection failed: {e}")
        print("Using fallback strategy: freeform optimizer")
        strategy_selection = None
    
    # Use selected optimizers
    if strategy_selection:
        selected_optimizers = strategy_selection.selected_optimizers
    
    # Step 3: Execute Optimizers
    print(f"\n⚡ Step 3: Executing optimizers concurrently...")
    try:
        execution_results = await executor.execute_optimizers(
            optimizer_names=selected_optimizers,
            context=context
        )
        
        print(f"✅ Execution completed:")
        print(f"   Total time: {execution_results.execution_time:.2f}s")
        print(f"   Successful: {execution_results.successful_count}")
        print(f"   Failed: {execution_results.failed_count}")
        
        # Display all results
        print(f"\n📊 All Results:")
        for i, result in enumerate(execution_results.results, 1):
            print(f"   {i}. {result.optimizer_name}:")
            print(f"      Status: {result.status}")
            print(f"      Confidence: {result.confidence:.2f}")
            print(f"      Execution time: {result.execution_time:.2f}s")
            if result.status.value == "failed":
                print(f"      Error: {result.error_message}")
            else:
                print(f"      Changes: {len(result.changes_made)} modifications")
                print(f"      Reasoning: {result.reasoning[:80]}...")
        
        # Display best result
        if execution_results.best_result:
            best = execution_results.best_result
            print(f"\n🏆 Best Result: {best.optimizer_name}")
            print(f"   Confidence: {best.confidence:.2f}")
            print(f"   Execution time: {best.execution_time:.2f}s")
            print(f"   Status: {best.status}")
            
            print(f"\n📝 Optimized Prompt:")
            print("-" * 50)
            print(best.candidate_prompt)
            print("-" * 50)
            
            print(f"\n🔍 Optimization Reasoning:")
            print(best.reasoning)
            
            print(f"\n📋 Changes Made:")
            for i, change in enumerate(best.changes_made, 1):
                print(f"   {i}. {change}")
            
            # Step 4: Update Context (optional for demo)
            print(f"\n🔄 Step 4: Updating context with results...")
            
            # Simulate new metrics (in real scenario, you'd evaluate the new prompt)
            
            with open("poc/metrics/enhanced_baseline_results_new.json", "r") as f:
                new_all_metrics = json.load(f)
            
            new_baseline_metrics = {k: v for k, v in new_all_metrics.items() if k != "detailed_failed_cases"}
            failed_cases = new_all_metrics.get("detailed_failed_cases", {}).get("wrong_classifications", [])
            failed_cases_summary = new_all_metrics.get("failed_cases_summary", {})
            
            updated_context = context_manager.update_context_with_results(
                current_context=context,
                new_prompt=best.candidate_prompt,
                new_metrics=new_baseline_metrics,
                optimizer_used=best.optimizer_name,
                new_failed_cases=failed_cases,
                new_failed_cases_summary=failed_cases_summary
            )
            
            print("New context: ", updated_context)
            
            print(f"✅ Context updated to iteration {updated_context.iteration_number}")
            
        else:
            print("❌ No successful optimization results")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Cleanup
    print(f"\n🧹 Step 5: Cleanup...")
    await claude_client.close()
    await context_manager.close()
    await orchestrator.close()
    await executor.close()
    
    print(f"\n✅ Optimization pipeline completed!")
    print("=" * 60)


def run_optimization_sync():
    """
    Synchronous wrapper to run the optimization pipeline
    """
    try:
        asyncio.run(run_optimization_pipeline())
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 Prompt Optimization System")
    print("Running complete pipeline from context creation to freeform execution")
    print()
    
    # Check if required files exist
    # required_files = [
    #     "poc/metrics/enhanced_baseline_results.json",
    #     "poc/intent_analysis/claude_intent_analysis_results.json",
    #     # "poc/metrics/enhanced_baseline_results_new.json"
    # ]
    
    # missing_files = []
    # for file_path in required_files:
    #     if not os.path.exists(file_path):
    #         missing_files.append(file_path)
    
    # if missing_files:
    #     print("❌ Missing required files:")
    #     for file_path in missing_files:
    #         print(f"   - {file_path}")
    #     print("\nPlease ensure these files exist before running the optimization.")
    #     sys.exit(1)
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY environment variable not set")
        print("   The system will attempt to run but may fail at API calls")
        print()
    
    # Run the pipeline
    run_optimization_sync() 