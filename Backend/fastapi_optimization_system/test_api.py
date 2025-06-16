#!/usr/bin/env python3
"""
Test script for the Auto Prompt Optimization FastAPI system
Demonstrates how to use the API with a sample request
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def test_optimization_api():
    """Test the complete optimization flow"""
    
    # Sample optimization request
    optimization_request = {
        "system_prompt": "You are a text classifier. Classify the input text according to the provided schema. Return your response as a valid JSON object with the specified fields.",
        "user_prompt": "Classify this text for sentiment and intent.",
        "schema": {
            "sentiment": ["positive", "negative", "neutral"],
            "intent": ["complaint", "feedback", "question", "request"]
        },
        "model_configuration": {
            "provider": "groq",
            "model_name": "llama-3.3-70b-versatile",
            "temperature": 0.2
        },
        "dataset": "text_classification_dataset",
        "max_iterations": 3,
        "improvement_threshold": 0.05,
        "enable_human_feedback": True
    }
    
    print("🚀 Testing Auto Prompt Optimization API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Service: {response.json()['service']}")
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test validation endpoint
    print("\n2. Testing request validation...")
    try:
        response = requests.post(
            f"{BASE_URL}/validate",
            json=optimization_request
        )
        if response.status_code == 200:
            validation_result = response.json()
            if validation_result["valid"]:
                print("✅ Request validation passed")
            else:
                print("❌ Request validation failed")
                print(f"   Errors: {validation_result['errors']}")
                return
        else:
            print("❌ Validation endpoint error")
            return
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return
    
    # Test optimization endpoint
    print("\n3. Starting optimization process...")
    try:
        response = requests.post(
            f"{BASE_URL}/optimize",
            json=optimization_request,
            timeout=1800  # 30 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Optimization completed successfully!")
            print(f"   Request ID: {result['request_id']}")
            print(f"   Status: {result['status']}")
            print(f"   Total iterations: {result['total_iterations']}")
            print(f"   Improvement: {result['improvement_percentage']:.2f}%")
            print(f"   Deployment recommendation: {result['deployment_recommendation']}")
            print(f"   Execution time: {result['total_execution_time']:.1f}s")
            
            # Print data split info
            data_split = result['data_split']
            print(f"\n📊 Data Split Summary:")
            print(f"   Total samples: {data_split['total_samples']}")
            print(f"   Train: {data_split['train_samples']} (25%)")
            print(f"   Dev A: {data_split['dev_a_samples']} (35%)")
            print(f"   Dev B: {data_split['dev_b_samples']} (20%)")
            print(f"   Test: {data_split['test_samples']} (20%)")
            
            # Print baseline vs optimized prompt comparison
            print(f"\n📝 Prompt Comparison:")
            print(f"   Baseline prompt length: {len(result['baseline_prompt'])} chars")
            print(f"   Optimized prompt length: {len(result['best_prompt'])} chars")
            
            # Print metrics summary
            if result['iterations_history']:
                print(f"\n🔄 Iteration History:")
                for iteration in result['iterations_history']:
                    print(f"   Iteration {iteration['iteration']}: {iteration['improvement_over_baseline']*100:.2f}% improvement")
            
            return result
            
        else:
            print(f"❌ Optimization failed with status {response.status_code}")
            print(f"   Error: {response.json()}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Optimization timed out (this is normal for long optimizations)")
        return None
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return None

def test_status_endpoint(request_id: str):
    """Test the status endpoint with a request ID"""
    print(f"\n4. Testing status endpoint for request: {request_id}")
    try:
        response = requests.get(f"{BASE_URL}/optimize/{request_id}/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ Status retrieved successfully")
            print(f"   Current step: {status['current_step']}")
            print(f"   Progress: {status['progress_percentage']:.1f}%")
            print(f"   Message: {status['message']}")
            if status['current_iteration']:
                print(f"   Iteration: {status['current_iteration']}/{status['total_iterations']}")
        else:
            print(f"❌ Status check failed: {response.json()}")
    except Exception as e:
        print(f"❌ Status check error: {e}")

def test_examples_endpoint():
    """Test the examples endpoint"""
    print("\n5. Testing examples endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/examples")
        if response.status_code == 200:
            examples = response.json()
            print("✅ Examples retrieved successfully")
            print(f"   Available examples: {list(examples['examples'].keys())}")
        else:
            print(f"❌ Examples endpoint failed")
    except Exception as e:
        print(f"❌ Examples error: {e}")

def test_models_endpoint():
    """Test the models endpoint"""
    print("\n6. Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()
            print("✅ Models list retrieved successfully")
            print(f"   Supported providers: {list(models['providers'].keys())}")
            print(f"   Recommended for development: {models['recommendations']['development']}")
        else:
            print(f"❌ Models endpoint failed")
    except Exception as e:
        print(f"❌ Models error: {e}")

if __name__ == "__main__":
    print("Auto Prompt Optimization API Test")
    print("Make sure the API server is running on http://localhost:8000")
    print("Start the server with: uvicorn main:app --reload")
    print()
    
    # Test all endpoints
    test_examples_endpoint()
    test_models_endpoint()
    
    # Main optimization test
    result = test_optimization_api()
    
    # Test status endpoint if we got a request ID
    if result and 'request_id' in result:
        test_status_endpoint(result['request_id'])
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    
    if result:
        print("\n💡 Key Takeaways:")
        print(f"   - Optimization improved performance by {result['improvement_percentage']:.2f}%")
        print(f"   - Recommendation: {result['deployment_recommendation']}")
        print(f"   - Process took {result['total_execution_time']:.1f} seconds")
        print(f"   - Used {result['total_iterations']} optimization iterations")
    else:
        print("\n⚠️  Note: Optimization may take several minutes to complete.")
        print("   Consider using the status endpoint to monitor progress.") 