import streamlit as st
import requests
import json
import time
import pandas as pd
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Prompt Optimization System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# API Base URL
API_BASE_URL = "http://localhost:8000/api/v1"

# Initialize session state
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'
    if 'uploaded_datasets' not in st.session_state:
        st.session_state.uploaded_datasets = []
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    if 'request_id' not in st.session_state:
        st.session_state.request_id = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'schema_fields' not in st.session_state:
        st.session_state.schema_fields = [{
            "key": "action", 
            "values": ["CODE_GENERATION", "NOT_FOUND"],
            "type": "simple",
            "nested_values": {},
            "parent_field": ""
        }]
    if 'upload_in_progress' not in st.session_state:
        st.session_state.upload_in_progress = False
    if 'upload_progress' not in st.session_state:
        st.session_state.upload_progress = {}
    if 'upload_start_time' not in st.session_state:
        st.session_state.upload_start_time = None
    if 'optimization_progress' not in st.session_state:
        st.session_state.optimization_progress = {}
    if 'last_progress_check' not in st.session_state:
        st.session_state.last_progress_check = None

# Helper functions
def make_api_call(endpoint, method="GET", data=None, files=None):
    """Make API calls with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        
        if response.status_code in [200, 201]:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def get_model_options():
    """Get available models"""
    models_data, error = make_api_call("/models")
    if error:
        return {"groq": ["llama-3.3-70b-versatile"], "anthropic": ["claude-3-sonnet"], "openai": ["gpt-4"], "google": ["gemini-pro"]}
    return models_data.get("providers", {"groq": ["llama-3.3-70b-versatile"], "anthropic": ["claude-3-sonnet"], "openai": ["gpt-4"], "google": ["gemini-pro"]})

def check_upload_progress():
    """Check upload progress and update session state"""
    if st.session_state.upload_in_progress and 'upload_request_id' in st.session_state:
        # In a real implementation, you'd check the upload status via an API
        # For now, we'll simulate progress tracking
        pass

def start_background_upload(files, data):
    """Start upload process and track progress"""
    st.session_state.upload_in_progress = True
    st.session_state.upload_start_time = datetime.now()
    st.session_state.upload_progress = {
        "dataset_name": data.get("dataset_name"),
        "status": "starting",
        "progress": 0,
        "message": "Initializing upload..."
    }
    
    # Make the actual API call
    result, error = make_api_call("/upload-dataset", "POST", data, files)
    
    if error:
        # Parse error details for better user experience
        error_details = parse_upload_error(error)
        st.session_state.upload_progress.update({
            "status": "failed",
            "error": error,
            "error_details": error_details,
            "message": f"Upload failed: {error_details['user_message']}"
        })
        st.session_state.upload_in_progress = False
        return None, error
    else:
        st.session_state.upload_progress.update({
            "status": "completed",
            "progress": 100,
            "message": "Upload completed successfully!",
            "result": result
        })
        st.session_state.upload_in_progress = False
        return result, None

def parse_upload_error(error_text):
    """Parse error text to extract meaningful information for users"""
    error_details = {
        "type": "unknown",
        "user_message": "An unexpected error occurred",
        "technical_details": error_text,
        "suggestions": []
    }
    
    # Check for common error patterns
    if "File size" in error_text and "exceeds maximum allowed size" in error_text:
        # Extract file sizes from error
        import re
        size_match = re.search(r'File size \((\d+) bytes\) exceeds maximum allowed size \((\d+) bytes\)', error_text)
        if size_match:
            actual_size = int(size_match.group(1))
            max_size = int(size_match.group(2))
            actual_mb = actual_size / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            
            error_details.update({
                "type": "file_size",
                "user_message": f"File too large: {actual_mb:.1f}MB (max: {max_mb:.1f}MB)",
                "suggestions": [
                    f"Reduce file size to under {max_mb:.0f}MB",
                    "Split large dataset into smaller files",
                    "Remove unnecessary columns or rows",
                    "Compress or optimize your CSV file"
                ]
            })
    
    elif "validation_error" in error_text.lower() or "422" in error_text:
        if "dataset_name" in error_text.lower():
            error_details.update({
                "type": "dataset_name",
                "user_message": "Invalid dataset name",
                "suggestions": [
                    "Use only letters, numbers, and underscores",
                    "Make dataset name at least 3 characters long",
                    "Avoid special characters like / \\ : * ? \" < > |"
                ]
            })
        elif "csv" in error_text.lower() or "format" in error_text.lower():
            error_details.update({
                "type": "file_format",
                "user_message": "Invalid file format",
                "suggestions": [
                    "Ensure file has .csv extension",
                    "Use UTF-8 encoding",
                    "Include 'input' and 'expected_output' columns",
                    "Check for proper CSV formatting"
                ]
            })
        else:
            error_details.update({
                "type": "validation",
                "user_message": "Validation failed",
                "suggestions": [
                    "Check file format and content",
                    "Verify dataset name is valid",
                    "Ensure all required fields are present"
                ]
            })
    
    elif "connection" in error_text.lower() or "timeout" in error_text.lower():
        error_details.update({
            "type": "connection",
            "user_message": "Connection error",
            "suggestions": [
                "Check your internet connection",
                "Try again in a few moments",
                "Reduce file size if upload keeps timing out"
            ]
        })
    
    elif "langfuse" in error_text.lower():
        error_details.update({
            "type": "langfuse",
            "user_message": "LangFuse service error",
            "suggestions": [
                "Check LangFuse API keys are configured",
                "Verify LangFuse service is accessible",
                "Try again in a few moments"
            ]
        })
    
    return error_details

def fetch_langfuse_datasets():
    """
    Fetch datasets from LangFuse via backend API
    """
    try:
        response = requests.get(f"{API_BASE_URL}/datasets", timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result.get('datasets', []), None
        else:
            error_detail = response.json() if response.content else {"message": f"HTTP {response.status_code}"}
            return [], error_detail
    except requests.exceptions.RequestException as e:
        return [], {"message": f"Connection error: {str(e)}"}
    except Exception as e:
        return [], {"message": f"Unexpected error: {str(e)}"}

def load_results_from_file(request_id):
    """
    Load optimization results directly from the intermediate_results directory
    This is a fallback when the API status endpoint fails
    """
    try:
        import os
        import json
        
        # Try to find the results directory
        results_dir = f"../Backend/fastapi_optimization_system/intermediate_results/{request_id}"
        
        # Look for final results file
        final_file = os.path.join(results_dir, "final", "optimization_results.json")
        if os.path.exists(final_file):
            with open(final_file, 'r') as f:
                results = json.load(f)
                return results, None
        
        # Look for any results file in the directory
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith('.json') and 'result' in file.lower():
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                results = json.load(f)
                                return results, None
                        except:
                            continue
        
        return None, "No results file found"
        
    except Exception as e:
        return None, f"Error reading results file: {str(e)}"

def check_optimization_progress():
    """
    Check optimization progress if one is running and update session state
    Includes throttling to prevent too frequent API calls and fallback to file reading
    """
    if st.session_state.optimization_running and st.session_state.request_id:
        # Throttle API calls - only check every 3 seconds
        current_time = datetime.now()
        if (st.session_state.last_progress_check and 
            (current_time - st.session_state.last_progress_check).seconds < 3):
            return False  # Too soon, don't check again
        
        try:
            # Get current status
            status_data, error = make_api_call(f"/optimize/{st.session_state.request_id}/status")
            
            if error:
                # If API fails, try to load results from file
                if "404" in str(error) or "Not Found" in str(error):
                    # Optimization might be completed but API endpoint is gone
                    file_results, file_error = load_results_from_file(st.session_state.request_id)
                    if file_results:
                        st.session_state.optimization_running = False
                        st.session_state.optimization_progress = {
                            "status": "completed",
                            "progress": 100,
                            "message": "Optimization completed successfully! (loaded from file)"
                        }
                        st.session_state.optimization_results = file_results
                        return True  # Completed
                
                st.session_state.optimization_progress = {
                    "status": "error",
                    "message": f"Error checking progress: {str(error)[:50]}...",
                    "progress": 0
                }
                st.session_state.optimization_running = False
                return False
            else:
                # Update progress in session state
                progress = status_data.get('progress_percentage', 0)
                current_status = status_data.get('status', 'running')
                
                st.session_state.optimization_progress = {
                    "status": "running",
                    "progress": progress,
                    "current_step": status_data.get('current_step', 'Processing...'),
                    "message": status_data.get('message', 'Processing...'),
                    "current_iteration": status_data.get('current_iteration'),
                    "total_iterations": status_data.get('total_iterations'),
                    "estimated_time_remaining": status_data.get('estimated_time_remaining')
                }
                

                
                st.session_state.last_progress_check = current_time
                
                # Check if completed
                if progress >= 100 or status_data.get('status') == 'completed':
                    st.session_state.optimization_running = False
                    st.session_state.optimization_progress["status"] = "completed"
                    
                    # Get final results - try API first, then file fallback
                    final_result, error = make_api_call(f"/optimize/{st.session_state.request_id}/status")
                    if not error and final_result.get('status') == 'completed':
                        st.session_state.optimization_results = final_result
                        st.session_state.optimization_progress["message"] = "Optimization completed successfully!"
                    else:
                        # Fallback to file reading
                        file_results, file_error = load_results_from_file(st.session_state.request_id)
                        if file_results:
                            st.session_state.optimization_results = file_results
                            st.session_state.optimization_progress["message"] = "Optimization completed successfully! (loaded from file)"
                    
                    return True  # Completed
                
                return False  # Still running
                
        except Exception as e:
            st.session_state.optimization_progress = {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "progress": 0
            }
            st.session_state.optimization_running = False
            return False
    
    return None  # No optimization running



# Main header
def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>Prompt Optimization System</h1>
        <p>Upload datasets, optimize prompts, and track performance with real-time feedback</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
def show_sidebar():
    with st.sidebar:
        st.title("📊 Navigation")
        
        # Navigation buttons
        if st.button("📁 Dataset Upload", use_container_width=True):
            st.session_state.current_page = 'upload'
        
        if st.button("⚙️ Configuration", use_container_width=True):
            st.session_state.current_page = 'config'
        
        if st.button("🔄 Optimization", use_container_width=True):
            st.session_state.current_page = 'optimization'
        
        if st.button("📈 Results", use_container_width=True):
            st.session_state.current_page = 'results'
        
        if st.button("📊 Reports", use_container_width=True):
            st.session_state.current_page = 'reports'
        
        st.divider()
        
        # Note: Progress checking is now handled globally in main()
        
        # Persistent status indicator
        if st.session_state.upload_in_progress:
            progress_data = st.session_state.upload_progress
            status = progress_data.get('status', 'starting')
            
            if status == 'starting':
                st.info("🔄 Uploading...")
                st.caption(f"{progress_data.get('dataset_name', 'Unknown')}")
                # Small progress bar in sidebar
                st.progress(0.3)
            elif status == 'failed':
                st.error("❌ Upload Failed")
                st.caption("Check main area for details")
                    
        elif st.session_state.optimization_running or st.session_state.optimization_progress.get('status') == 'running':
            # Show optimization progress
            progress_data = st.session_state.optimization_progress
            progress_value = progress_data.get('progress', 0) / 100
            
            st.info("🔄 Optimization Running")
            st.progress(progress_value)
            
            # Progress details
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Progress", f"{progress_data.get('progress', 0):.0f}%")
            with col_p2:
                if progress_data.get('current_iteration') and progress_data.get('total_iterations'):
                    st.metric("Iteration", f"{progress_data['current_iteration']}/{progress_data['total_iterations']}")
                else:
                    st.metric("Status", "Processing")
            
            # Current step
            current_step = progress_data.get('current_step', 'Processing...')
            if len(current_step) > 25:
                current_step = current_step[:22] + "..."
            st.caption(f"📍 {current_step}")
            
            # Quick action buttons
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if st.button("📈 View Details", key="sidebar_view_details", use_container_width=True):
                    st.session_state.current_page = 'optimization'
                    st.rerun()
            with col_a2:
                if st.button("⏹️ Cancel", key="sidebar_cancel", use_container_width=True):
                    cancel_result, error = make_api_call(f"/optimize/{st.session_state.request_id}", "DELETE")
                    st.session_state.optimization_running = False
                    st.session_state.request_id = None
                    st.session_state.optimization_progress = {}
                    st.warning("Optimization cancelled")
                    st.rerun()
        
        elif st.session_state.optimization_progress.get('status') == 'completed':
            st.success("✅ Optimization Complete")
            if st.button("📈 View Results", key="sidebar_view_results", use_container_width=True):
                st.session_state.current_page = 'results'
                st.rerun()
        
        elif st.session_state.optimization_progress.get('status') == 'error':
            st.error("❌ Optimization Error")
            error_msg = st.session_state.optimization_progress.get('message', 'Unknown error')
            if len(error_msg) > 30:
                error_msg = error_msg[:27] + "..."
            st.caption(error_msg)
            
            # Add manual load button if we have a request_id
            if st.session_state.request_id:
                if st.button("🔄 Try Load Results", key="sidebar_manual_load", use_container_width=True):
                    file_results, file_error = load_results_from_file(st.session_state.request_id)
                    if file_results:
                        st.session_state.optimization_results = file_results
                        st.session_state.optimization_progress = {
                            "status": "completed",
                            "progress": 100,
                            "message": "Results loaded successfully from file!"
                        }
                        st.success("✅ Results loaded!")
                        st.rerun()
                    else:
                        st.error(f"Failed to load: {file_error}")
        
        elif st.session_state.optimization_results:
            st.success("✅ Results Available")
            improvement = st.session_state.optimization_results.get('improvement_percentage', 0)
            st.metric("Improvement", f"{improvement:.1f}%")
            if st.button("📈 View Results", key="sidebar_view_final_results", use_container_width=True):
                st.session_state.current_page = 'results'
                st.rerun()
        
        st.divider()

# Page 1: Dataset Upload
def show_dataset_upload():
    st.header("📁 Dataset Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload CSV Dataset")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with 'input' and 'expected_output' columns"
        )
        
        # Dataset details
        dataset_name = st.text_input(
            "Dataset Name", 
            placeholder="e.g., code_generation_dataset",
            help="Unique name for your dataset"
        )
        
        description = st.text_area(
            "Description (Optional)",
            placeholder="Brief description of your dataset..."
        )
        
        # Upload button
        if st.button("🚀 Upload Dataset", type="primary", use_container_width=True):
            if uploaded_file and dataset_name:
                if not st.session_state.upload_in_progress:
                    # Set upload state immediately to show progress bar
                    st.session_state.upload_in_progress = True
                    st.session_state.upload_start_time = datetime.now()
                    st.session_state.upload_progress = {
                        "dataset_name": dataset_name,
                        "status": "starting",
                        "message": "Starting upload..."
                    }
                    
                    # Show immediate progress feedback
                    with st.spinner("Starting upload..."):
                        files = {"file": uploaded_file}
                        data = {
                            "dataset_name": dataset_name,
                            "description": description
                        }
                        
                        # Start upload process
                        result, error = start_background_upload(files, data)
                        
                        if error:
                            st.error(f"❌ Upload failed: {error}")
                            # Clear upload state on error
                            st.session_state.upload_in_progress = False
                        else:
                            st.success("✅ Dataset uploaded successfully!")
                            st.session_state.uploaded_datasets.append({
                                "name": dataset_name,
                                "description": description,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "stats": result.get("upload_results", {})
                            })
                            # Clear upload state on success
                            st.session_state.upload_in_progress = False
                    
                    # Force rerun to show updated state
                    st.rerun()
                else:
                    st.warning("⚠️ Upload already in progress. Please wait for current upload to complete.")
            else:
                st.error("Please provide both file and dataset name")
        
        # Show current upload progress on the upload page
        if st.session_state.upload_in_progress:
            st.divider()
            st.subheader("🔄 Upload Progress")
            
            progress_data = st.session_state.upload_progress
            
            # Progress info
            col_up1, col_up2 = st.columns(2)
            with col_up1:
                st.metric("Dataset", progress_data.get('dataset_name', 'Unknown'))
            with col_up2:
                elapsed_time = ""
                if st.session_state.upload_start_time:
                    elapsed = datetime.now() - st.session_state.upload_start_time
                    elapsed_time = f"{elapsed.seconds}s"
                st.metric("Elapsed Time", elapsed_time)
            
            # Always show progress bar when upload is in progress
            st.progress(0.5, "🔄 Uploading to LangFuse...")
            
            # Status message
            status = progress_data.get('status', 'starting')
            if status == 'starting':
                st.info("Upload in progress... You can navigate to other pages while this completes.")
            elif status == 'completed':
                st.progress(1.0, "Upload completed!")
                st.success("✅ Upload completed successfully!")
            elif status == 'failed':
                error_details = progress_data.get('error_details', {})
                st.error(f"❌ Upload failed: {error_details.get('user_message', 'Unknown error')}")
                
                # Show actionable suggestions
                suggestions = error_details.get('suggestions', [])
                if suggestions:
                    st.markdown("**💡 How to fix this:**")
                    for suggestion in suggestions:
                        st.markdown(f"• {suggestion}")
                
                # Technical details in expander for developers
                with st.expander("🔧 Technical Details"):
                    st.code(error_details.get('technical_details', 'No details available'))
                
            # Cancel upload button
            if st.button("❌ Cancel Upload"):
                st.session_state.upload_in_progress = False
                st.session_state.upload_progress = {}
                st.warning("Upload cancelled")
                st.rerun()
    
    with col2:
        st.subheader("📋 CSV Format Example")
        example_data = {
            "input": [
                "Generate a React component for a login form",
                "Create a Flutter widget for navigation",
                "Fix this JavaScript error: undefined variable"
            ],
            "expected_output": [
                '{"action": "CODE_GENERATION", "platform": "DYNAMIC_WEB_APPLICATION", "framework": "REACT"}',
                '{"action": "CODE_GENERATION", "platform": "DYNAMIC_MOBILE_APP", "framework": "FLUTTER"}',
                '{"action": "NOT_FOUND", "subAction": "ERROR", "platform": "NOT_FOUND"}'
            ]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)

# Page 2: Configuration
def show_configuration():
    st.header("⚙️ Optimization Configuration")
    
    # Schema builder (outside form to allow buttons)
    st.subheader("Schema Configuration")
    st.markdown("""
    Define the JSON structure and possible values for classification. **Now supports nested fields!**
    """)

    
    
    # Schema management buttons
    col_schema1, col_schema2, col_schema3 = st.columns([1, 1, 3])
    with col_schema1:
        if st.button("➕ Add Field", use_container_width=True):
            st.session_state.schema_fields.append({
                "key": "", 
                "values": [], 
                "type": "simple",
                "nested_values": {},
                "parent_field": ""
            })
            st.rerun()
    
    with col_schema2:
        if st.button("📋 Load Example", use_container_width=True):
            # Load the educational example schema
            st.session_state.schema_fields = [
                {
                    "key": "action",
                    "values":  ["CODE_GENERATION", "NOT_FOUND"],
                    "type": "simple",
                    "nested_values": {},
                    "parent_field": ""
                },
                {
                    "key": "subAction",
                    "values": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
                    "type": "simple",
                    "nested_values": {},
                    "parent_field": ""
                },
                {
                    "key": "platform",
                    "values": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION",
                        "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
                    "type": "simple",
                    "nested_values": {},
                    "parent_field": ""
                },
                {
                    "key": "framework",
                    "values": ["REACT", "FLUTTER", "NOT_FOUND"],
                    "type": "simple",
                    "nested_values": {},
                    "parent_field": ""
                },
                {
                    "key": "languageType",
                    "values": ["REACT_JAVASCRIPT", "NOT_FOUND"],
                    "type": "simple",
                    "nested_values": {},
                    "parent_field": ""
                }
                
            ]
            st.rerun()
    
    # Display existing fields
    for i, field in enumerate(st.session_state.schema_fields):
        with st.container():
            # Field header
            col_header1, col_header2 = st.columns([4, 1])
            with col_header1:
                st.markdown(f"**Field {i+1}:** `{field.get('key', 'Unnamed')}`")
            with col_header2:
                if st.button("🗑️", key=f"delete_{i}", help="Delete field"):
                    st.session_state.schema_fields.pop(i)
                    st.rerun()
            
            # Field configuration
            col_a, col_b, col_c = st.columns([2, 2, 2])
            
            with col_a:
                new_key = st.text_input(
                    "Field Name", 
                    value=field.get('key', ''), 
                    key=f"key_{i}",
                    help="e.g., intent, subject, topic"
                )
                field['key'] = new_key
            
            with col_b:
                field_type = st.selectbox(
                    "Field Type",
                    ["simple", "nested"],
                    index=0 if field.get('type', 'simple') == 'simple' else 1,
                    key=f"type_{i}",
                    help="Simple: fixed values, Nested: values depend on another field"
                )
                field['type'] = field_type
            
            with col_c:
                if field_type == "nested":
                    # Get available parent fields (only simple fields that come before this one)
                    parent_options = [""]
                    for j, parent_field in enumerate(st.session_state.schema_fields[:i]):
                        if parent_field.get('type') == 'simple' and parent_field.get('key'):
                            parent_options.append(parent_field['key'])
                    
                    parent_field = st.selectbox(
                        "Parent Field",
                        parent_options,
                        index=parent_options.index(field.get('parent_field', '')) if field.get('parent_field', '') in parent_options else 0,
                        key=f"parent_{i}",
                        help="Field whose values determine this field's options"
                    )
                    field['parent_field'] = parent_field
                else:
                    st.write("") # Empty space to align
            
            # Values configuration
            if field_type == "simple":
                values_str = ", ".join(field.get('values', [])) if field.get('values') else ""
                new_values = st.text_area(
                    f"Possible Values (comma-separated)",
                    value=values_str,
                    key=f"values_{i}",
                    height=80,
                    help="e.g., CONCEPT_EXPLANATION, PROBLEM_SOLVING, NOT_FOUND"
                )
                field['values'] = [v.strip() for v in new_values.split(",") if v.strip()]
                field['nested_values'] = {}  # Clear nested values for simple fields
                
            else:  # nested type
                st.write("**Nested Values Configuration:**")
                if field.get('parent_field'):
                    # Get parent field values
                    parent_field_obj = None
                    for parent in st.session_state.schema_fields:
                        if parent.get('key') == field['parent_field']:
                            parent_field_obj = parent
                            break
                    
                    if parent_field_obj and parent_field_obj.get('values'):
                        nested_values = field.get('nested_values', {})
                        
                        # Create tabs for each parent value
                        parent_values = parent_field_obj['values']
                        if len(parent_values) <= 5:  # Use tabs for small number of values
                            tabs = st.tabs([f"📋 {val}" for val in parent_values if val != "NOT_FOUND"])
                            
                            for idx, parent_val in enumerate([v for v in parent_values if v != "NOT_FOUND"]):
                                with tabs[idx]:
                                    current_nested = ", ".join(nested_values.get(parent_val, []))
                                    new_nested = st.text_area(
                                        f"Values for {parent_val}",
                                        value=current_nested,
                                        key=f"nested_{i}_{parent_val}",
                                        height=70,
                                        help=f"Values when {field['parent_field']} = {parent_val}"
                                    )
                                    nested_values[parent_val] = [v.strip() for v in new_nested.split(",") if v.strip()]
                        else:  # Use expanders for many values
                            for parent_val in parent_values:
                                if parent_val != "NOT_FOUND":
                                    with st.expander(f"📋 {parent_val}"):
                                        current_nested = ", ".join(nested_values.get(parent_val, []))
                                        new_nested = st.text_area(
                                            f"Values for {parent_val}",
                                            value=current_nested,
                                            key=f"nested_{i}_{parent_val}",
                                            height=70,
                                            help=f"Values when {field['parent_field']} = {parent_val}"
                                        )
                                        nested_values[parent_val] = [v.strip() for v in new_nested.split(",") if v.strip()]
                        
                        field['nested_values'] = nested_values
                        field['values'] = []  # Clear simple values for nested fields
                    else:
                        st.warning(f"⚠️ Parent field '{field.get('parent_field', '')}' has no values defined.")
                else:
                    st.warning("⚠️ Please select a parent field for nested configuration.")
            
            st.divider()
    
    st.divider()
    
    # Model configuration (outside form for reactivity)
    st.subheader("🤖 Model Configuration")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        provider = st.selectbox(
            "Provider",
            ["groq", "anthropic", "openai", "google"],
            index=0,
            help="AI model provider"
        )
    
    with col_m2:
        # Hardcoded model options for reliability
        model_options = {
            "groq": ["llama-3.3-70b-versatile"],
            "anthropic": ["claude-sonnet-4-20250514"],
            "openai": ["o4-mini-2025-04-16"],
            "google": ["gemini-2.5-pro-preview-06-05"]
        }
        available_models = model_options.get(provider, ["llama-3.3-70b-versatile"])
        model_name = st.selectbox("Model", available_models)
    
    with col_m3:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    st.divider()
    
    # Configuration form
    with st.form("optimization_config"):
        # Prompts section
        st.subheader("📝 Prompts")
        
        col1, col2 = st.columns(2)
        with col1:
            system_prompt = st.text_area(
                "System Prompt *",
                value="You are a classification model. Classify the input into the correct category. Return the result in JSON format.",
                height=150,
                help="The main instruction for the AI model"
            )
        
        with col2:
            user_prompt = st.text_area(
                "User Prompt (Optional)",
                placeholder="Additional context or instructions...",
                height=150,
                help="Additional context that will be added to each query"
            )
        
        # Dataset selection
        st.subheader("📊 Dataset Selection")
        
        # Fetch datasets from LangFuse
        with st.spinner("Loading datasets from LangFuse..."):
            langfuse_datasets, error = fetch_langfuse_datasets()
        
        if error:
            st.error(f"❌ Failed to load datasets from LangFuse: {error.get('message', 'Unknown error')}")
            st.warning("⚠️ Falling back to manual dataset entry.")
            selected_dataset = st.text_input("Dataset Name", placeholder="Enter dataset name manually")
        elif langfuse_datasets:
            # Display datasets from LangFuse
            dataset_options = []
            dataset_info = {}
            
            for dataset in langfuse_datasets:
                name = dataset.get('name', 'Unknown')
                dataset_options.append(name)
                dataset_info[name] = {
                    'description': dataset.get('description', 'No description'),
                    'created_at': dataset.get('createdAt', 'Unknown'),
                    'id': dataset.get('id', 'Unknown')
                }
            
            selected_dataset = st.selectbox(
                "Select Dataset from LangFuse",
                dataset_options,
                help="Datasets fetched from your LangFuse project"
            )
        else:
            st.info("📂 No datasets found in LangFuse. Upload a dataset first or enter manually.")
            selected_dataset = st.text_input("Dataset Name", placeholder="Enter dataset name manually")
        
        # Submit button
        submitted = st.form_submit_button("Start Optimization", type="primary", use_container_width=True)
        
        if submitted:
            # Validate inputs
            if not system_prompt.strip():
                st.error("System prompt is required")
                return
            
            if not selected_dataset:
                st.error("Please select or specify a dataset")
                return
            
            # Build schema
            schema = {}
            for field in st.session_state.schema_fields:
                if field.get('key'):
                    if field.get('type') == 'nested' and field.get('nested_values'):
                        # For nested fields, use the nested_values structure
                        schema[field['key']] = field['nested_values']
                    elif field.get('type') == 'simple' and field.get('values'):
                        # For simple fields, use the flat values list
                        schema[field['key']] = field['values']
            
            if not schema:
                st.error("Please add at least one schema field")
                return
            
            # Prepare optimization request
            optimization_request = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt if user_prompt.strip() else "Classify the following:",
                "schema": schema,
                "model_configuration": {
                    "provider": provider,
                    "model_name": model_name,
                    "temperature": temperature
                },
                "dataset": selected_dataset,
                "max_iterations": 5,  # Default value
                "improvement_threshold": 0.05,  # Default value
                "enable_human_feedback": True  # Default value
            }
            
            # Store config and start optimization
            st.session_state.current_optimization_config = optimization_request
            st.session_state.current_page = 'optimization'
            st.rerun()
    
    # Show current schema preview
    if st.session_state.schema_fields:
        st.subheader("📋 Schema Preview")
        schema_preview = {}
        for field in st.session_state.schema_fields:
            if field.get('key'):
                if field.get('type') == 'nested' and field.get('nested_values'):
                    # For nested fields, show the nested structure
                    schema_preview[field['key']] = field['nested_values']
                elif field.get('type') == 'simple' and field.get('values'):
                    # For simple fields, show the flat values list
                    schema_preview[field['key']] = field['values']
        
        if schema_preview:
            st.json(schema_preview)
        else:
            st.info("Add field names and values to see schema preview")

# Page 3: Optimization Progress
def show_optimization():
    st.header("🔄 Optimization Progress")
    
    if not hasattr(st.session_state, 'current_optimization_config'):
        st.warning("⚠️ No optimization configuration found. Please configure first.")
        if st.button("Go to Configuration"):
            st.session_state.current_page = 'config'
            st.rerun()
        return
    
    # Start optimization if not running
    if not st.session_state.optimization_running and not st.session_state.request_id:
        if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
            with st.spinner("Starting optimization..."):
                result, error = make_api_call("/optimize", "POST", st.session_state.current_optimization_config)
                
                if error:
                    st.error(f"❌ Failed to start optimization: {error}")
                else:
                    st.session_state.request_id = result.get('request_id')
                    st.session_state.optimization_running = True
                    st.success("✅ Optimization started!")
                    st.rerun()
    
    # Show progress if optimization is running
    if st.session_state.optimization_running and st.session_state.request_id:
        # Use centralized progress checking
        progress_completed = check_optimization_progress()
        
        if st.session_state.optimization_progress.get('status') == 'error':
            st.error(f"❌ Error: {st.session_state.optimization_progress.get('message', 'Unknown error')}")
            st.session_state.optimization_running = False
        
        else:
            progress_data = st.session_state.optimization_progress
            
            # Progress bar
            progress = progress_data.get('progress', 0)
            st.progress(progress / 100)
            
            # Status info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Progress", f"{progress:.1f}%")
            
            with col2:
                if progress_data.get('current_iteration') and progress_data.get('total_iterations'):
                    st.metric("Iteration", f"{progress_data['current_iteration']}/{progress_data['total_iterations']}")
                else:
                    st.metric("Status", "Processing...")
            
            with col3:
                if progress_data.get('estimated_time_remaining'):
                    st.metric("ETA", f"{progress_data['estimated_time_remaining']}s")
                else:
                    st.metric("ETA", "Calculating...")
            
            # Current step
            st.info(f"**Current Step:** {progress_data.get('current_step', 'Unknown')}")
            st.write(f"**Message:** {progress_data.get('message', 'Processing...')}")
            
            # Human feedback notification
            if 'human_feedback' in progress_data.get('current_step', '').lower():
                st.warning("🔍 **Human Feedback Required**")
                st.write("The system is collecting human feedback via LangFuse. The annotation interface will open automatically.")
                
                if st.button("🌐 Open LangFuse", help="Click if LangFuse didn't open automatically"):
                    st.write("Opening LangFuse annotation interface...")
                    st.markdown("[Click here to open LangFuse](https://cloud.langfuse.com)")
            
            # Check if completed
            if progress_completed:
                st.success("🎉 Optimization completed successfully!")
                st.balloons()
                
                if st.button("📈 View Results"):
                    st.session_state.current_page = 'results'
                    st.rerun()
            # Note: Auto-refresh is now handled globally in main()
        
        # Cancel button
        if st.button("❌ Cancel Optimization"):
            cancel_result, error = make_api_call(f"/optimize/{st.session_state.request_id}", "DELETE")
            st.session_state.optimization_running = False
            st.session_state.request_id = None
            st.session_state.optimization_progress = {}
            st.warning("Optimization cancelled")
            st.rerun()
    
    # Show completion status if optimization finished
    elif st.session_state.optimization_progress.get('status') == 'completed':
        st.success("🎉 Optimization completed successfully!")
        st.info("📈 Results are ready to view!")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if st.button("📈 View Results", type="primary", use_container_width=True):
                st.session_state.current_page = 'results'
                st.rerun()
        with col_res2:
            if st.button("🔄 Start New Optimization", use_container_width=True):
                st.session_state.current_page = 'config'
                st.rerun()

# Page 4: Results
def load_available_results():
    """
    Load list of available optimization results from the backend directory
    """
    try:
        import os
        results_base_dir = "../Backend/fastapi_optimization_system/intermediate_results"
        
        if not os.path.exists(results_base_dir):
            return []
        
        available_results = []
        for item in os.listdir(results_base_dir):
            item_path = os.path.join(results_base_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory has results
                final_dir = os.path.join(item_path, "final")
                if os.path.exists(final_dir):
                    # Get creation time
                    try:
                        creation_time = os.path.getctime(final_dir)
                        available_results.append({
                            "request_id": item,
                            "path": item_path,
                            "created": datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
                        })
                    except:
                        available_results.append({
                            "request_id": item,
                            "path": item_path,
                            "created": "Unknown"
                        })
        
        # Sort by creation time (newest first)
        available_results.sort(key=lambda x: x["created"], reverse=True)
        return available_results
    
    except Exception as e:
        st.error(f"Error loading available results: {str(e)}")
        return []

def show_results():
    st.header("📈 Optimization Results")
    
    if not st.session_state.optimization_results:
        st.warning("⚠️ No optimization results available.")
        
        # Show available results that can be loaded
        st.subheader("📂 Load Previous Results")
        available_results = load_available_results()
        
        if available_results:
            st.write("Found these completed optimizations:")
            
            for result in available_results[:5]:  # Show last 5
                col_r1, col_r2, col_r3 = st.columns([3, 2, 1])
                
                with col_r1:
                    st.write(f"**ID:** `{result['request_id'][:8]}...`")
                
                with col_r2:
                    st.write(f"**Created:** {result['created']}")
                
                with col_r3:
                    if st.button("📥 Load", key=f"load_{result['request_id'][:8]}"):
                        file_results, file_error = load_results_from_file(result['request_id'])
                        if file_results:
                            st.session_state.optimization_results = file_results
                            st.session_state.request_id = result['request_id']
                            st.success("✅ Results loaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to load results: {file_error}")
        else:
            st.info("💡 No completed optimizations found. Run an optimization first!")
        
        return
    
    results = st.session_state.optimization_results
    
    # Key metrics
    st.subheader("🎯 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        improvement = results.get('improvement_percentage', 0)
        st.metric(
            "Overall Improvement", 
            f"{improvement:.1f}%",
            delta=f"{improvement:.1f}%" if improvement > 0 else None
        )
    
    with col2:
        accuracy = results.get('best_metrics', {}).get('overall_accuracy', 0)
        st.metric("Final Accuracy", f"{accuracy:.3f}")
    
    with col3:
        iterations = results.get('total_iterations', 0)
        st.metric("Iterations Completed", iterations)
    
    with col4:
        recommendation = results.get('deployment_recommendation', 'unknown')
        color = "🟢" if recommendation == "deploy" else "🟡"
        st.metric("Recommendation", f"{color} {recommendation.title()}")
    
    # Optimized prompt
    st.subheader("✨ Optimized Prompt")
    st.code(results.get('best_prompt', 'No prompt available'), language="text")
    
    # Comparison with baseline
    st.subheader("📊 Baseline vs Optimized Comparison")
    
    baseline_metrics = results.get('baseline_metrics', {})
    best_metrics = results.get('best_metrics', {})
    
    if baseline_metrics and best_metrics:
        comparison_data = {
            'Metric': ['Overall Accuracy', 'Valid JSON Rate', 'Average F1 Score'],
            'Baseline': [
                baseline_metrics.get('overall_accuracy', 0),
                baseline_metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0),
                baseline_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
            ],
            'Optimized': [
                best_metrics.get('overall_accuracy', 0),
                best_metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0),
                best_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        df['Improvement'] = df['Optimized'] - df['Baseline']
        
        st.dataframe(df.style.format({
            'Baseline': '{:.3f}',
            'Optimized': '{:.3f}',
            'Improvement': '{:+.3f}'
        }), use_container_width=True)
        
        # Visualization
        fig = px.bar(
            df, 
            x='Metric', 
            y=['Baseline', 'Optimized'],
            title='Performance Comparison',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Iteration history
    if results.get('iterations_history'):
        st.subheader("🔄 Optimization History")
        
        for i, iteration in enumerate(results['iterations_history']):
            with st.expander(f"Iteration {iteration.get('iteration', i+1)} - {iteration.get('optimizer_used', 'Unknown')}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Improvement", f"{iteration.get('improvement_over_baseline', 0):+.3f}")
                    st.metric("Confidence", f"{iteration.get('confidence', 0):.2f}")
                
                with col_b:
                    st.write("**Reasoning:**")
                    st.write(iteration.get('reasoning', 'No reasoning provided'))
                
                st.write("**Generated Prompt:**")
                st.code(iteration.get('candidate_prompt', 'No prompt available'), language="text")
    
    # Human feedback summary
    if results.get('human_feedback_summary'):
        st.subheader("👥 Human Feedback Summary")
        feedback = results['human_feedback_summary']
        
        # Handle both dict and string feedback data
        if isinstance(feedback, str):
            # If it's a string, try to parse as JSON or display as text
            try:
                feedback = json.loads(feedback)
            except (json.JSONDecodeError, TypeError):
                st.write(f"**Feedback Summary:** {feedback}")
                feedback = {}  # Set to empty dict to prevent further errors
        
        if isinstance(feedback, dict):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.metric("Cases Reviewed", feedback.get('total_cases_reviewed', feedback.get('total_cases', 0)))
            
            with col_f2:
                st.metric("Accuracy Improvement", f"{feedback.get('accuracy_improvement', 0):.1f}%")
            
            with col_f3:
                insights = feedback.get('key_insights', feedback.get('key_feedback_themes', []))
                st.metric("Key Insights", len(insights) if isinstance(insights, list) else 0)
            
            if insights and isinstance(insights, list):
                st.write("**Key Insights:**")
                for insight in insights:
                    st.write(f"• {insight}")
            
            problematic_fields = feedback.get('problematic_fields', feedback.get('improvement_suggestions', []))
            if problematic_fields and isinstance(problematic_fields, list):
                st.write("**Areas for Improvement:**")
                for field in problematic_fields:
                    st.write(f"• {field}")
        else:
            # Fallback for other types
            st.write(f"**Feedback Summary:** {str(feedback)}")

# Page 5: Reports
def show_reports():
    st.header("📊 Detailed Reports")
    
    if not st.session_state.optimization_results:
        st.warning("⚠️ No optimization results available. Please run an optimization first.")
        return
    
    results = st.session_state.optimization_results
    
    # Performance trends
    st.subheader("📈 Performance Trends")
    
    if results.get('iterations_history'):
        iterations_data = []
        for iteration in results['iterations_history']:
            iterations_data.append({
                'Iteration': iteration.get('iteration', 0),
                'Improvement': iteration.get('improvement_over_baseline', 0),
                'Optimizer': iteration.get('optimizer_used', 'Unknown'),
                'Confidence': iteration.get('confidence', 0)
            })
        
        df_iterations = pd.DataFrame(iterations_data)
        
        # Line chart for improvements
        fig_line = px.line(
            df_iterations,
            x='Iteration',
            y='Improvement',
            title='Improvement Over Iterations',
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Bar chart for confidence
        fig_conf = px.bar(
            df_iterations,
            x='Iteration',
            y='Confidence',
            color='Optimizer',
            title='Confidence Scores by Optimizer'
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detailed metrics breakdown
    st.subheader("🔍 Detailed Metrics Analysis")
    
    baseline_metrics = results.get('baseline_metrics', {})
    test_metrics = results.get('test_metrics', {})
    
    if baseline_metrics and test_metrics:
        # Field-wise performance
        baseline_fields = baseline_metrics.get('field_metrics', {})
        test_fields = test_metrics.get('field_metrics', {})
        
        if baseline_fields and test_fields:
            field_comparison = []
            for field in baseline_fields.keys():
                if field in test_fields:
                    baseline_acc = baseline_fields[field].get('accuracy', 0)
                    test_acc = test_fields[field].get('accuracy', 0)
                    field_comparison.append({
                        'Field': field,
                        'Baseline Accuracy': baseline_acc,
                        'Final Accuracy': test_acc,
                        'Improvement': test_acc - baseline_acc
                    })
            
            if field_comparison:
                df_fields = pd.DataFrame(field_comparison)
                
                fig_fields = px.bar(
                    df_fields,
                    x='Field',
                    y=['Baseline Accuracy', 'Final Accuracy'],
                    title='Field-wise Performance Comparison',
                    barmode='group'
                )
                st.plotly_chart(fig_fields, use_container_width=True)
                
                st.dataframe(df_fields.style.format({
                    'Baseline Accuracy': '{:.3f}',
                    'Final Accuracy': '{:.3f}',
                    'Improvement': '{:+.3f}'
                }), use_container_width=True)
    
    # Export options
    st.subheader("💾 Export Results")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        if st.button("📄 Export as JSON", use_container_width=True):
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col_e2:
        if st.button("📊 Export Summary CSV", use_container_width=True):
            summary_data = {
                'Metric': ['Overall Improvement', 'Final Accuracy', 'Iterations', 'Recommendation'],
                'Value': [
                    f"{results.get('improvement_percentage', 0):.1f}%",
                    f"{results.get('best_metrics', {}).get('overall_accuracy', 0):.3f}",
                    results.get('total_iterations', 0),
                    results.get('deployment_recommendation', 'unknown')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Main app
def main():
    init_session_state()
    show_header()
    
    # Global optimization progress tracking - works on all pages
    if st.session_state.optimization_running:
        # Add auto-refresh for all pages when optimization is running
        progress_completed = check_optimization_progress()
        if progress_completed:
            st.balloons()
            st.success("🎉 **Optimization completed!** Results are ready to view.")
        else:
            # Use a gentle refresh mechanism to prevent overwhelming the app
            time.sleep(2)  # Small delay to prevent too rapid refreshing
            st.rerun()
    
    # Check for optimization completion notification
    if st.session_state.optimization_progress.get('status') == 'completed' and st.session_state.current_page != 'optimization':
        st.success("🎉 **Optimization completed successfully!** Click 'View Results' in the sidebar or go to the Results page.")
    
    # Only show non-blocking status messages for errors or brief success
    if st.session_state.upload_in_progress:
        progress_data = st.session_state.upload_progress
        
        # Only show status for failures or brief success notification
        if progress_data.get('status') == 'failed':
            error_details = progress_data.get('error_details', {})
            st.error(f"❌ Upload failed: {error_details.get('user_message', 'Unknown error')}")
            
            # Show error suggestions in expander
            suggestions = error_details.get('suggestions', [])
            if suggestions:
                with st.expander("💡 How to fix this"):
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
                        
        elif progress_data.get('status') == 'completed':
            st.success("🎉 Dataset uploaded successfully!")
            
            # Auto-clear completed status and add to datasets
            if 'result' in progress_data:
                result = progress_data['result']
                new_dataset = {
                    "name": progress_data['dataset_name'],
                    "description": result.get("description", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "stats": result.get("upload_results", {})
                }
                # Check if dataset already exists
                dataset_exists = any(d['name'] == new_dataset['name'] for d in st.session_state.uploaded_datasets)
                if not dataset_exists:
                    st.session_state.uploaded_datasets.append(new_dataset)
            
            # Clear the progress state automatically after showing success
            st.session_state.upload_progress = {}
            st.session_state.upload_in_progress = False
    
    show_sidebar()
    
    # Route to appropriate page
    if st.session_state.current_page == 'upload':
        show_dataset_upload()
    elif st.session_state.current_page == 'config':
        show_configuration()
    elif st.session_state.current_page == 'optimization':
        show_optimization()
    elif st.session_state.current_page == 'results':
        show_results()
    elif st.session_state.current_page == 'reports':
        show_reports()

if __name__ == "__main__":
    main() 