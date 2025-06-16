#!/usr/bin/env python3
"""
Test script for the Dataset Upload API
Demonstrates how to upload CSV datasets to LangFuse via the FastAPI system
"""

import requests
import json
import io
import csv
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def create_sample_csv() -> io.StringIO:
    """Create a sample CSV dataset for testing"""
    
    sample_data = [
        {
            "input": "I love this product! It works perfectly and exceeded my expectations.",
            "expected_output": '{"sentiment": "positive", "intent": "feedback"}',
            "category": "product_review",
            "source": "customer_feedback"
        },
        {
            "input": "This is terrible, I want a refund immediately.",
            "expected_output": '{"sentiment": "negative", "intent": "complaint"}',
            "category": "product_review", 
            "source": "customer_feedback"
        },
        {
            "input": "How do I use this feature? I can't find the documentation.",
            "expected_output": '{"sentiment": "neutral", "intent": "question"}',
            "category": "support_query",
            "source": "support_ticket"
        },
        {
            "input": "Can you please help me with my order status?",
            "expected_output": '{"sentiment": "neutral", "intent": "request"}',
            "category": "support_query",
            "source": "support_ticket"
        },
        {
            "input": "Amazing service! Thank you so much for the quick response.",
            "expected_output": '{"sentiment": "positive", "intent": "feedback"}',
            "category": "service_review",
            "source": "customer_feedback"
        }
    ]
    
    # Create CSV content
    csv_buffer = io.StringIO()
    fieldnames = ["input", "expected_output", "category", "source"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in sample_data:
        writer.writerow(row)
    
    csv_buffer.seek(0)
    return csv_buffer

def test_dataset_upload_api():
    """Test the complete dataset upload flow"""
    
    print("🚀 Testing Dataset Upload API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing dataset upload service health check...")
    try:
        response = requests.get(f"{BASE_URL}/dataset-upload-status")
        if response.status_code == 200:
            print("✅ Dataset upload service health check passed")
            health_data = response.json()
            print(f"   Service: {health_data['service']}")
            print(f"   Max file size: {health_data['limits']['max_file_size_mb']}MB")
            print(f"   Max items: {health_data['limits']['max_items_per_dataset']}")
        else:
            print("❌ Dataset upload service health check failed")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test examples endpoint
    print("\n2. Testing dataset upload examples...")
    try:
        response = requests.get(f"{BASE_URL}/dataset-upload-examples")
        if response.status_code == 200:
            print("✅ Examples retrieved successfully")
            examples = response.json()
            print(f"   Available CSV examples: {list(examples['csv_examples'].keys())}")
            print(f"   API request examples: {list(examples['api_request_examples'].keys())}")
        else:
            print("❌ Examples endpoint failed")
    except Exception as e:
        print(f"❌ Examples error: {e}")
    
    # Test validation endpoint
    print("\n3. Testing request validation...")
    validation_request = {
        "dataset_name": "test_classification_dataset",
        "description": "Test dataset for classification",
        "metadata": {
            "version": "1.0",
            "source": "test_script"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/validate-dataset-upload",
            json=validation_request
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
    
    # Test dataset upload
    print("\n4. Testing CSV dataset upload...")
    try:
        # Create sample CSV
        csv_content = create_sample_csv()
        csv_bytes = csv_content.getvalue().encode('utf-8')
        
        # Prepare form data
        files = {
            'file': ('test_dataset.csv', csv_bytes, 'text/csv')
        }
        
        data = {
            'dataset_name': 'test_classification_dataset',
            'description': 'Test dataset for text classification uploaded via API',
            'metadata': json.dumps({
                "version": "1.0",
                "source": "test_script",
                "created_by": "api_test",
                "total_examples": 5
            })
        }
        
        response = requests.post(
            f"{BASE_URL}/upload-dataset",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Dataset upload completed successfully!")
            print(f"   Request ID: {result['request_id']}")
            print(f"   Dataset name: {result['dataset_name']}")
            print(f"   File size: {result['file_info']['size_bytes']} bytes")
            
            # Print upload statistics
            upload_stats = result['upload_results']
            print(f"\n📊 Upload Statistics:")
            print(f"   Total items: {upload_stats['total_items']}")
            print(f"   Successful: {upload_stats['successful']}")
            print(f"   Failed: {upload_stats['failed']}")
            
            if upload_stats['failed'] > 0:
                print(f"   Errors: {len(upload_stats.get('errors', []))}")
            
            # Test getting dataset info
            print("\n5. Testing dataset info retrieval...")
            dataset_name = result['dataset_name']
            info_response = requests.get(f"{BASE_URL}/dataset/{dataset_name}")
            
            if info_response.status_code == 200:
                info_result = info_response.json()
                print("✅ Dataset info retrieved successfully")
                dataset_info = info_result['dataset_info']
                print(f"   Dataset ID: {dataset_info.get('id', 'N/A')}")
                print(f"   Name: {dataset_info.get('name', 'N/A')}")
                print(f"   Description: {dataset_info.get('description', 'N/A')}")
            else:
                print(f"❌ Failed to get dataset info: {info_response.status_code}")
                print(f"   Error: {info_response.json()}")
            
            return result
            
        else:
            print(f"❌ Dataset upload failed with status {response.status_code}")
            error_detail = response.json()
            print(f"   Error: {error_detail}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Dataset upload timed out")
        return None
    except Exception as e:
        print(f"❌ Dataset upload error: {e}")
        return None

def test_error_cases():
    """Test various error cases"""
    print("\n" + "=" * 50)
    print("🧪 Testing Error Cases")
    print("=" * 50)
    
    # Test invalid file type
    print("1. Testing invalid file type...")
    try:
        files = {
            'file': ('test.txt', b'not a csv file', 'text/plain')
        }
        data = {
            'dataset_name': 'test_invalid_file'
        }
        
        response = requests.post(
            f"{BASE_URL}/upload-dataset",
            files=files,
            data=data
        )
        
        if response.status_code == 422:
            print("✅ Invalid file type correctly rejected")
            error = response.json()
            print(f"   Error: {error['detail']['message']}")
        else:
            print(f"❌ Expected 422, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing invalid file type: {e}")
    
    # Test invalid dataset name
    print("\n2. Testing invalid dataset name...")
    try:
        csv_content = create_sample_csv()
        csv_bytes = csv_content.getvalue().encode('utf-8')
        
        files = {
            'file': ('test.csv', csv_bytes, 'text/csv')
        }
        data = {
            'dataset_name': 'a'  # Too short
        }
        
        response = requests.post(
            f"{BASE_URL}/upload-dataset",
            files=files,
            data=data
        )
        
        if response.status_code == 422:
            print("✅ Invalid dataset name correctly rejected")
            error = response.json()
            print(f"   Error: {error['detail']['message']}")
        else:
            print(f"❌ Expected 422, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing invalid dataset name: {e}")
    
    # Test invalid metadata JSON
    print("\n3. Testing invalid metadata JSON...")
    try:
        csv_content = create_sample_csv()
        csv_bytes = csv_content.getvalue().encode('utf-8')
        
        files = {
            'file': ('test.csv', csv_bytes, 'text/csv')
        }
        data = {
            'dataset_name': 'test_invalid_metadata',
            'metadata': 'invalid json string'
        }
        
        response = requests.post(
            f"{BASE_URL}/upload-dataset",
            files=files,
            data=data
        )
        
        if response.status_code == 422:
            print("✅ Invalid metadata JSON correctly rejected")
            error = response.json()
            print(f"   Error: {error['detail']['message']}")
        else:
            print(f"❌ Expected 422, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing invalid metadata: {e}")

if __name__ == "__main__":
    print("Dataset Upload API Test")
    print("Make sure the API server is running on http://localhost:8000")
    print("Start the server with: uvicorn main:app --reload")
    print()
    
    # Test main functionality
    result = test_dataset_upload_api()
    
    # Test error cases
    test_error_cases()
    
    print("\n" + "=" * 50)
    print("🎉 Dataset upload API testing completed!")
    
    if result:
        print("\n💡 Key Takeaways:")
        print(f"   - Successfully uploaded {result['upload_results']['successful']} items")
        print(f"   - Dataset created: {result['dataset_name']}")
        print(f"   - File size: {result['file_info']['size_bytes']} bytes")
        print(f"   - Request ID: {result['request_id']}")
        print("\n📝 Next Steps:")
        print("   - Use the uploaded dataset in prompt optimization")
        print("   - Check LangFuse dashboard to verify the dataset")
        print("   - Try uploading your own CSV files")
    else:
        print("\n⚠️  Note: Dataset upload may require proper LangFuse configuration.")
        print("   Make sure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.") 