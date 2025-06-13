"""
Test Human Feedback Integration with LangFuse

This script demonstrates how to:
1. Set up human feedback workflow
2. Create evaluation traces
3. Add traces to annotation queues
4. Collect human feedback

Run this after setting up LangFuse credentials.
"""

import asyncio
import json
import os
from datetime import datetime
from prompt_optimizer.core.human_feedback import (
    HumanFeedbackManager, 
    DevBHumanFeedbackIntegration,
    get_default_human_feedback_config
)

async def test_human_feedback_setup():
    """Test the human feedback setup process."""
    print("🧪 Testing Human Feedback Integration Setup")
    print("=" * 50)
    
    # Initialize components
    feedback_manager = HumanFeedbackManager()
    feedback_integration = DevBHumanFeedbackIntegration(feedback_manager)
    
    # Test 1: Setup workflow
    print("\n1️⃣ Testing workflow setup...")
    setup_success = feedback_integration.setup_human_feedback_workflow()
    print(f"   Setup success: {setup_success}")
    
    return feedback_manager, feedback_integration

async def test_trace_creation():
    """Test creating evaluation traces for human review."""
    print("\n2️⃣ Testing trace creation...")
    
    feedback_manager = HumanFeedbackManager()
    
    # Sample Dev B evaluation results
    sample_dev_b_results = [
        {
            "evaluation_id": "test_eval_1",
            "input_prompt": "Classify this code request: Create a React component for user authentication",
            "model_output": '{"action": "create", "subAction": "component", "platform": "web", "framework": "react", "languageType": "javascript"}',
            "expected_output": {
                "action": "create",
                "subAction": "component", 
                "platform": "web",
                "framework": "react",
                "languageType": "javascript"
            },
            "failure_type": "correct_classification"
        },
        {
            "evaluation_id": "test_eval_2", 
            "input_prompt": "Classify this code request: Debug Python Flask API endpoint",
            "model_output": '{"action": "debug", "subAction": "api", "platform": "backend", "framework": "django", "languageType": "python"}',
            "expected_output": {
                "action": "debug",
                "subAction": "api",
                "platform": "backend", 
                "framework": "flask",
                "languageType": "python"
            },
            "failure_type": "wrong_classification",
            "wrong_fields": ["framework"]
        }
    ]
    
    # Sample candidate prompt and baseline metrics
    candidate_prompt = """
    You are a code classification expert. Classify the following code request into these categories:
    
    Action: create, modify, debug, explain, optimize
    SubAction: component, function, api, database, test
    Platform: web, mobile, backend, desktop
    Framework: react, vue, angular, flask, django, express
    LanguageType: javascript, python, java, csharp, go
    
    Respond with valid JSON only.
    """
    
    baseline_metrics = {
        "summary": {
            "average_enum_macro_f1": 0.75,
            "total_examples": 100
        }
    }
    
    # Create traces
    trace_ids = feedback_manager.batch_create_evaluation_traces(
        evaluation_results=sample_dev_b_results,
        candidate_prompt=candidate_prompt,
        baseline_metrics=baseline_metrics
    )
    
    print(f"   Created {len(trace_ids)} traces")
    for i, trace_id in enumerate(trace_ids):
        print(f"     Trace {i+1}: {trace_id}")
    
    return trace_ids

async def test_feedback_collection(trace_ids):
    """Test collecting human feedback (simulation)."""
    print("\n3️⃣ Testing feedback collection...")
    
    feedback_manager = HumanFeedbackManager()
    
    # Test getting feedback summary (will be empty initially)
    feedback_summary = feedback_manager.get_human_feedback_summary(trace_ids)
    
    print(f"   Total traces: {feedback_summary['total_traces']}")
    print(f"   Annotated traces: {feedback_summary['annotated_traces']}")
    print(f"   Pending traces: {feedback_summary['pending_traces']}")
    
    return feedback_summary

async def test_complete_workflow():
    """Test the complete human feedback workflow."""
    print("\n4️⃣ Testing complete workflow integration...")
    
    feedback_manager = HumanFeedbackManager()
    feedback_integration = DevBHumanFeedbackIntegration(feedback_manager)
    
    # Sample data for complete workflow test
    sample_dev_b_results = [
        {
            "evaluation_id": "workflow_test_1",
            "input_prompt": "Create a Vue.js component for data visualization",
            "model_output": '{"action": "create", "subAction": "component", "platform": "web", "framework": "vue", "languageType": "javascript"}',
            "expected_output": {
                "action": "create",
                "subAction": "component",
                "platform": "web", 
                "framework": "vue",
                "languageType": "javascript"
            }
        }
    ]
    
    candidate_prompt = "Optimized classification prompt for code requests..."
    baseline_metrics = {"summary": {"average_enum_macro_f1": 0.72}}
    
    # Process results for human feedback
    trace_ids = feedback_integration.process_dev_b_results_for_human_feedback(
        dev_b_results=sample_dev_b_results,
        candidate_prompt=candidate_prompt,
        baseline_metrics=baseline_metrics
    )
    
    print(f"   Workflow generated {len(trace_ids)} traces for review")
    
    return trace_ids

def print_manual_setup_instructions():
    """Print detailed manual setup instructions."""
    print("\n📋 MANUAL SETUP INSTRUCTIONS")
    print("=" * 50)
    
    config = get_default_human_feedback_config()
    
    print("\n🔧 LangFuse Dashboard Setup:")
    print("1. Go to your LangFuse dashboard")
    print("2. Navigate to 'Settings' > 'Score Configs'")
    print("3. Create the following score configurations:")
    
    for score_config in config.score_configs:
        print(f"\n   📊 {score_config['name']}:")
        print(f"      Type: {score_config['type']}")
        print(f"      Description: {score_config['description']}")
        if 'categories' in score_config:
            print(f"      Categories: {', '.join(score_config['categories'])}")
    
    print(f"\n4. Navigate to 'Annotate' tab")
    print(f"5. Click '+ New annotation queue'")
    print(f"6. Configure queue:")
    print(f"   Name: {config.queue_name}")
    print(f"   Description: {config.description}")
    print(f"   Reviewers per run: {config.reviewers_per_run}")
    print(f"   Enable reservations: {config.enable_reservations}")
    
    print(f"\n🔄 Workflow Process:")
    print("1. Run optimization cycle")
    print("2. System creates evaluation traces")
    print("3. Manually add traces to annotation queue in LangFuse")
    print("4. Complete human annotations")
    print("5. System retrieves feedback results")

async def main():
    """Main test function."""
    print("🚀 Human Feedback Integration Test Suite")
    print("=" * 60)
    
    # Check environment variables
    required_env_vars = ['LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your .env file")
        return
    
    try:
        # Test 1: Setup
        feedback_manager, feedback_integration = await test_human_feedback_setup()
        
        # Test 2: Trace creation
        trace_ids = await test_trace_creation()
        
        # Test 3: Feedback collection
        feedback_summary = await test_feedback_collection(trace_ids)
        
        # Test 4: Complete workflow
        workflow_trace_ids = await test_complete_workflow()
        
        # Print manual instructions
        print_manual_setup_instructions()
        
        print(f"\n✅ All tests completed successfully!")
        print(f"   Created {len(trace_ids + workflow_trace_ids)} total traces")
        print(f"   Next: Complete manual setup in LangFuse dashboard")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 