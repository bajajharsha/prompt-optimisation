import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Prompt Optimization Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .progress-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .timer-display {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Base URL
API_BASE_URL = "http://localhost:8000/api/v1"



# Initialize session state
def init_session_state():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    if 'request_id' not in st.session_state:
        st.session_state.request_id = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'dataset_uploaded' not in st.session_state:
        st.session_state.dataset_uploaded = False
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'schema_fields' not in st.session_state:
        st.session_state.schema_fields = {}
    if 'optimization_start_time' not in st.session_state:
        st.session_state.optimization_start_time = None

def make_api_call(endpoint, method="GET", data=None, files=None):
    """Make API calls with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        # Use longer timeout for results endpoint
        timeout = 30
        if '/results' in endpoint:
            timeout = 120  # 2 minutes for large result files
        elif method == "POST":
            if files:
                timeout = 60
            else:
                timeout = 300
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=timeout)
            else:
                response = requests.post(url, json=data, timeout=timeout)
        
        if response.status_code in [200, 201]:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        return None, f"Request timeout after {timeout} seconds. Large optimization results may take time to load - please try again."
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def format_elapsed_time(start_time):
    """Format elapsed time since start_time"""
    if not start_time:
        return "N/A"
    
    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>Prompt Optimization Dashboard</h1>
        <p>Optimize your prompts with automated testing and human feedback</p>
    </div>
    """, unsafe_allow_html=True)

def show_navigation():
    """Simple horizontal navigation"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("1. Upload Dataset", key="nav_upload"):
            st.session_state.current_step = 'upload'
            st.rerun()
    
    with col2:
        if st.button("2. Configuration", key="nav_config"):
            st.session_state.current_step = 'config'
            st.rerun()
    
    with col3:
        if st.button("Past Reports", key="nav_reports"):
            st.session_state.current_step = 'reports'
            st.rerun()

# Step 1: Dataset Upload
def show_upload_step():
    st.header("Step 1: Upload Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV file with 'input' and 'expected_output' columns"
        )
        
        dataset_name = st.text_input(
            "Dataset Name", 
            placeholder="e.g., text_classification_dataset"
        )
        
        if st.button("Upload Dataset", type="primary"):
            if uploaded_file and dataset_name:
                with st.spinner("Uploading dataset..."):
                    files = {"file": uploaded_file}
                    data = {"dataset_name": dataset_name}
                    
                    result, error = make_api_call("/upload-dataset", "POST", data, files)
                    
                    if error:
                        st.error(f"Upload failed: {error}")
                    else:
                        st.success("Dataset uploaded successfully!")
                        st.session_state.dataset_uploaded = True
                        st.session_state.selected_dataset = dataset_name
                        st.session_state.current_step = 'config'
                        st.rerun()
            else:
                st.error("Please provide both file and dataset name")
    
    with col2:
        st.subheader("CSV Format")
        example_df = pd.DataFrame({
            "input": ["Classify this text", "Another example"],
            "expected_output": ['{"intent": "question"}', '{"intent": "statement"}']
        })
        st.dataframe(example_df, use_container_width=True)
        
        if st.session_state.dataset_uploaded:
            st.success(f"Dataset ready: {st.session_state.selected_dataset}")

# Step 2: Configuration
def show_config_step():
    st.header("Step 2: Configuration")
    

    
    # Check if optimization is running and show progress
    if st.session_state.optimization_running:
        st.subheader("🚀 Optimization in Progress")
        st.info("Your optimization is running. Please wait for completion...")
        
        # Check status
        completed = check_optimization_status()
        
        if completed:
            st.balloons()
            st.success("Optimization completed! Redirecting to reports...")
            time.sleep(2)
            st.rerun()
            return
        
        # Get current status
        status_data, error = make_api_call(f"/optimize/{st.session_state.request_id}/status")
        
        if not error and status_data:
            # Ensure status_data is a dictionary
            if not isinstance(status_data, dict):
                st.error(f"Invalid status data format: {type(status_data)}")
                return
                
            progress = status_data.get('progress_percentage', 0)
            
            # Progress bar
            st.progress(progress / 100)
            
            # Status info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Progress", f"{progress:.0f}%")
            
            with col2:
                current_iter = status_data.get('current_iteration')
                total_iter = status_data.get('total_iterations')
                if current_iter and total_iter:
                    st.metric("Iteration", f"{current_iter}/{total_iter}")
                else:
                    st.metric("Status", "Processing")
            
            # Current step
            current_step = status_data.get('current_step', 'Processing...')
            st.info(f"Current Step: {current_step}")
            
            # Auto-refresh every 3 seconds
            time.sleep(3)
            st.rerun()
        else:
            st.error(f"Failed to get status: {error}")
            st.info("Retrying in 5 seconds...")
            time.sleep(5)
            st.rerun()
        
        return  # Don't show configuration options while optimization is running
    
    # Main content area (only shown when not optimizing)
    main_content = st.container()
    
    with main_content:
        # Dataset selection
        st.subheader("Dataset Selection")
        
        # Fetch available datasets
        with st.spinner("Loading available datasets..."):
            datasets_data, error = make_api_call("/datasets")
            
            if error:
                st.error(f"Failed to load datasets: {error}")
                st.info("Please upload a dataset in Step 1 first.")
                selected_dataset = None
            else:
                datasets = datasets_data.get('datasets', [])
                if datasets:
                    dataset_names = [d.get('name', 'Unknown') for d in datasets]
                    
                    # Set default selection to uploaded dataset if available
                    default_index = 0
                    if st.session_state.dataset_uploaded and st.session_state.selected_dataset in dataset_names:
                        default_index = dataset_names.index(st.session_state.selected_dataset)
                    
                    selected_dataset = st.selectbox(
                        "Select a dataset:",
                        options=dataset_names,
                        index=default_index,
                        help="Choose from available datasets"
                    )
                    
                    # Show dataset info
                    if selected_dataset:
                        selected_dataset_info = next((d for d in datasets if d.get('name') == selected_dataset), None)
                        if selected_dataset_info:
                            with st.expander("📊 Dataset Information", expanded=False):
                                st.write(f"**Name:** {selected_dataset_info.get('name', 'N/A')}")
                                if selected_dataset_info.get('description'):
                                    st.write(f"**Description:** {selected_dataset_info.get('description')}")
                                st.write(f"**Created:** {selected_dataset_info.get('createdAt', 'N/A')[:10]}")
                else:
                    st.warning("No datasets available. Please upload a dataset in Step 1 first.")
                    selected_dataset = None
        
        st.divider()
        
        # Only proceed if we have a selected dataset
        if selected_dataset:
            # Schema configuration section
            st.subheader("🔧 Schema Definition")
            st.caption("Define the JSON structure for classification output")
            
            # Tab selection for schema building
            tab1, tab2 = st.tabs(["Build Custom", "Use Example"])
            
            with tab1:
                st.markdown("**Build Your Own Schema:**")
                
                # Initialize schema fields directly in session state if not present
                if 'schema_fields' not in st.session_state:
                    st.session_state.schema_fields = {}
                
                # Add new field interface - more compact
                with st.form("add_field_form", clear_on_submit=True):
                    col_name, col_values, col_add = st.columns([2, 3, 1])
                    
                    with col_name:
                        new_field_name = st.text_input(
                            "Field Name", 
                            placeholder="e.g., intent, user.category",
                            help="Supports nested fields with dots"
                        )
                    
                    with col_values:
                        new_field_values = st.text_input(
                            "Values (comma-separated)",
                            placeholder="e.g., positive, negative, neutral",
                            help="Enter all possible values separated by commas"
                        )
                    
                    with col_add:
                        st.write("")  # Space for alignment
                        submitted = st.form_submit_button("➕ Add", type="primary", use_container_width=True)
                    
                    if submitted:
                        if new_field_name and new_field_values:
                            # Validate field name format
                            parts = new_field_name.split('.')
                            if len(parts) > 2:
                                st.error("Maximum 2 levels supported (parent.field)")
                            else:
                                # Add directly to schema_fields
                                st.session_state.schema_fields[new_field_name] = new_field_values
                                st.success(f"Added field: {new_field_name}")
                                st.rerun()
                        else:
                            st.error("Please fill in both field name and values")
                
                # Display current schema fields with inline editing
                if st.session_state.schema_fields:
                    st.markdown("**Current Schema Fields:**")
                    
                    fields_to_delete = []
                    
                    for field_name, field_values in st.session_state.schema_fields.items():
                        col_field, col_values, col_actions = st.columns([1.5, 3, 1])
                        
                        with col_field:
                            st.write(f"**{field_name}**")
                        
                        with col_values:
                            # Inline editing - handle different data types properly
                            if isinstance(field_values, (list, dict)):
                                # For complex data types, show as JSON and don't allow direct editing
                                st.json(field_values)
                                st.caption("🔒 Complex field (use Education button to modify)")
                            else:
                                # For string values, allow direct editing
                                updated_values = st.text_input(
                                    "Values:",
                                    value=str(field_values),
                                    key=f"edit_{field_name}",
                                    label_visibility="collapsed"
                                )
                                # Auto-update when changed
                                if updated_values != str(field_values):
                                    st.session_state.schema_fields[field_name] = updated_values
                        
                        with col_actions:
                            if st.button("🗑️", key=f"delete_{field_name}", help="Remove field"):
                                fields_to_delete.append(field_name)
                    
                    # Delete fields marked for deletion
                    for field_name in fields_to_delete:
                        del st.session_state.schema_fields[field_name]
                        st.rerun()
                    
                    # Quick actions
                    col_clear, col_space = st.columns([1, 3])
                    with col_clear:
                        if st.button("Clear All Fields", type="secondary"):
                            st.session_state.schema_fields = {}
                            st.rerun()
            
            with tab2:
                st.markdown("**Quick Start with Examples:**")
                col_ex1, col_ex2 = st.columns(2)
                
                with col_ex1:
                    if st.button("Code Generation", type="secondary"):
                        st.session_state.schema_fields = {
                            "action": "CODE_GENERATION, NOT_FOUND",
                            "subAction": "CODING, VISUAL_EDITS, ERROR, GENERAL",
                            "platform": "DYNAMIC_WEB_APPLICATION, STATIC_WEB_APPLICATION, DYNAMIC_MOBILE_APP, STATIC_MOBILE_APP, NOT_FOUND",
                            "framework": "REACT, FLUTTER, NOT_FOUND",
                            "languageType": "REACT_JAVASCRIPT, NOT_FOUND"
                        }
                        st.rerun()
                with col_ex2:
                    if st.button("Education", type="secondary"):
                        st.session_state.schema_fields = {
                            "intent": [
                                "CONCEPT_EXPLANATION", "PROBLEM_SOLVING", "MCQ_PRACTICE", 
                                "THEORY_REVIEW", "REAL_WORLD_APPLICATION", "EXAM_PREPARATION", "NOT_FOUND"
                            ],
                            "subject": [
                                "MATH", "PHYSICS", "CHEMISTRY", "BIOLOGY", "HISTORY", 
                                "GEOGRAPHY", "ENGLISH", "COMPUTER_SCIENCE", "NOT_FOUND"
                            ],
                            "topic": {
                                "MATH": ["ALGEBRA", "GEOMETRY", "TRIGONOMETRY", "CALCULUS", "STATISTICS", "NUMBER_SYSTEMS", "NOT_FOUND"],
                                "PHYSICS": ["LAWS_OF_MOTION", "GRAVITATION", "WORK_AND_ENERGY", "OPTICS", "THERMODYNAMICS", "ELECTRICITY", "NOT_FOUND"],
                                "CHEMISTRY": ["ATOMIC_STRUCTURE", "CHEMICAL_REACTIONS", "PERIODIC_TABLE", "ACIDS_BASES_SALTS", "METALS_NONMETALS", "NOT_FOUND"],
                                "BIOLOGY": ["CELL_STRUCTURE", "HUMAN_BODY", "PLANT_PHYSIOLOGY", "HEREDITY_AND_EVOLUTION", "MICROORGANISMS", "NOT_FOUND"],
                                "HISTORY": ["ANCIENT_CIVILIZATIONS", "WORLD_WARS", "FREEDOM_MOVEMENTS", "MEDIEVAL_HISTORY", "MODERN_HISTORY", "NOT_FOUND"],
                                "GEOGRAPHY": ["WEATHER_AND_CLIMATE", "PHYSICAL_FEATURES", "RESOURCES", "ENVIRONMENTAL_STUDIES", "NOT_FOUND"],
                                "ENGLISH": ["GRAMMAR", "COMPREHENSION", "LITERATURE", "WRITING_SKILLS", "VOCABULARY", "NOT_FOUND"],
                                "COMPUTER_SCIENCE": ["PROGRAMMING_BASICS", "ALGORITHMS", "DATA_STRUCTURES", "CYBER_SECURITY", "NOT_FOUND"],
                                "NOT_FOUND": ["NOT_FOUND"]
                            },
                            "difficulty": ["EASY", "MEDIUM", "HARD", "NOT_FOUND"],
                            "gradeLevel": [
                                "GRADE_6", "GRADE_7", "GRADE_8", "GRADE_9", 
                                "GRADE_10", "GRADE_11", "GRADE_12", "NOT_FOUND"
                            ]
                        }
                        st.rerun()

            
            # Generate schema for optimization
            schema = {}
            if st.session_state.get('schema_fields'):
                for field_name, field_values in st.session_state.schema_fields.items():
                    # Handle different types of field values
                    if isinstance(field_values, dict):
                        # Already a dictionary (e.g., nested topic structure)
                        schema[field_name] = field_values
                    elif isinstance(field_values, list):
                        # Already a list (e.g., intent, subject arrays)
                        schema[field_name] = field_values
                    elif isinstance(field_values, str) and field_values.strip():
                        # String that needs to be processed
                        # Handle nested schemas
                        if '.' in field_name:
                            parts = field_name.split('.')
                            parent, child = parts[0], parts[1]
                            if parent not in schema:
                                schema[parent] = {}
                            schema[parent][child] = [v.strip() for v in field_values.split(',') if v.strip()]
                        else:
                            schema[field_name] = [v.strip() for v in field_values.split(',') if v.strip()]
            
            st.divider()
            
            # Prompts section
            st.subheader("💬 Prompts")
            st.caption("Configure the system and user prompts for the model")
            
            col_sys, col_user = st.columns(2)
            with col_sys:
                system_prompt = st.text_area(
                    "System Prompt",
                    value="You are a json classifier. Classify the input according to the schema and return valid JSON.",
                    height=120,
                    help="Instructions for the AI model"
                )
            
            with col_user:
                user_prompt = st.text_input(
                    "User Prompt Template",
                    value="Classify this text:",
                    help="Template for user messages"
                )
                st.write("")  # Spacing to align with text area
                
            st.divider()
            
            # Model selection section
            st.subheader("🤖 Model Configuration")
            st.caption("Choose the AI model provider and specific model")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                provider = st.selectbox("Provider", ["groq", "anthropic", "openai", "google"])
            
            with col_m2:
                model_options = {
                    "groq": ["llama-3.3-70b-versatile"],
                    "anthropic": ["claude-sonnet-4-20250514"], 
                    "openai": ["gpt-4.1-mini-2025-04-14"],
                    "google": ["gemini-2.5-pro-preview-06-05"]
                }
                model_name = st.selectbox("Model", options=model_options[provider])
            
            st.divider()
            
            # Optimization controls section  
            st.subheader("🚀 Start Optimization")
            st.caption("Review your configuration and start the optimization process")
            
            # Summary of configuration
            with st.expander("📋 Configuration Summary", expanded=False):
                col_sum1, col_sum2 = st.columns(2)
                with col_sum1:
                    st.write(f"**Dataset:** {selected_dataset}")
                    st.write(f"**Schema Fields:** {len(schema)} fields")
                    st.write(f"**Model:** {provider} / {model_name}")
                with col_sum2:
                    if schema:
                        st.write("**Schema Preview:**")
                        st.json(schema)
            
            # Start optimization button
            if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
                if schema and system_prompt:
                    # Set timer and state IMMEDIATELY when button is clicked
                    st.session_state.optimization_running = True
                    st.session_state.optimization_start_time = datetime.now()
                    st.session_state.request_id = "temp-" + str(int(time.time()))  # Temporary ID
                    
                    st.success("✅ Timer started! Optimization beginning...")
                    
                    optimization_config = {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "schema": schema,
                        "model_configuration": {
                            "provider": provider,
                            "model_name": model_name,
                            "temperature": 0.2
                        },
                        "dataset": selected_dataset,
                        "max_iterations": 3,
                        "improvement_threshold": 0.05,
                        "enable_human_feedback": True
                    }
                    
                    # Start optimization
                    with st.spinner("Starting optimization..."):
                        result, error = make_api_call("/optimize", "POST", optimization_config)
                        
                        if error:
                            st.error(f"Failed to start optimization: {error}")
                            # Reset state on error
                            st.session_state.optimization_running = False
                            st.session_state.optimization_start_time = None
                            st.session_state.request_id = None
                        else:
                            # Update with real request ID
                            st.session_state.request_id = result.get('request_id')
                            st.success(f"Optimization started! Request ID: {st.session_state.request_id}")
                    
                    # Force page rerun to show timer
                            st.rerun()
                elif not schema:
                    st.error("Please add at least one schema field with values")
                else:
                    st.error("Please fill in all required fields")
        else:
            # Show helpful message when no dataset is selected
            st.info("👆 Please select a dataset above to continue with configuration")

# Step 3: Optimization Progress
def check_optimization_status():
    """Check optimization status"""
    if st.session_state.optimization_running and st.session_state.request_id:
        status_data, error = make_api_call(f"/optimize/{st.session_state.request_id}/status")
        
        if error:
            # Try loading from file
            results_data, results_error = make_api_call(f"/optimize/{st.session_state.request_id}/results")
            if not results_error:
                # Calculate total time taken
                if st.session_state.optimization_start_time:
                    st.session_state.optimization_duration = format_elapsed_time(st.session_state.optimization_start_time)
                else:
                    st.session_state.optimization_duration = "Unknown"
                
                st.session_state.optimization_running = False
                st.session_state.optimization_results = results_data
                st.session_state.optimization_start_time = None  # Reset timer
                st.session_state.current_step = 'reports'
                return True
            return False
        
        progress = status_data.get('progress_percentage', 0)
        
        if progress >= 100 or status_data.get('status') == 'completed':
            # Load final results
            results_data, results_error = make_api_call(f"/optimize/{st.session_state.request_id}/results")
            if not results_error:
                # Calculate total time taken
                if st.session_state.optimization_start_time:
                    st.session_state.optimization_duration = format_elapsed_time(st.session_state.optimization_start_time)
                else:
                    st.session_state.optimization_duration = "Unknown"
                
                st.session_state.optimization_running = False
                st.session_state.optimization_results = results_data
                st.session_state.optimization_start_time = None  # Reset timer
                st.session_state.current_step = 'reports'
                return True
        
        return False
    return None



# Results Page
def show_results():
    st.header("📊 Comprehensive Optimization Analysis")
    
    if not st.session_state.optimization_results:
        st.warning("No optimization results available.")
        return
    
    # Load from complete results structure - check both nested and direct access
    if 'data' in st.session_state.optimization_results:
        data = st.session_state.optimization_results.get('data', {})
    else:
        data = st.session_state.optimization_results
    
    # Extract all available metrics
    final_results = data.get('final_results', {})
    train_baseline_metrics = data.get('train_baseline_metrics', {})
    dev_a_baseline_metrics = data.get('dev_a_baseline_metrics', {})
    optimization_results = data.get('optimization_results', {})
    # The final_metrics in optimization_results IS the Dev A optimized metrics
    dev_a_optimized_metrics = optimization_results.get('final_metrics', {})
    test_metrics = data.get('test_metrics', {})
    iterations_data = data.get('iterations', [])
    
    # === TOP LEVEL SUMMARY ===
    st.subheader("Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Overall improvement (convert to percentage) - now using Dev A metrics
        if dev_a_baseline_metrics and dev_a_optimized_metrics:
            baseline_acc = dev_a_baseline_metrics.get('overall_accuracy', 0)
            optimized_acc = dev_a_optimized_metrics.get('overall_accuracy', 0)
            improvement = (optimized_acc - baseline_acc) * 100
        else:
            improvement = final_results.get('dev_a_improvement', final_results.get('dev_a_accuracy_improvement', 0)) * 100
        st.metric("Dev A Improvement", f"{improvement:.1f}%", 
                 delta=f"{improvement:+.1f}%" if improvement != 0 else None)
    
    with col2:
        # Final accuracy (convert to percentage) - now using Dev A metrics
        if dev_a_optimized_metrics and dev_a_baseline_metrics:
            final_accuracy = dev_a_optimized_metrics.get('overall_accuracy', 0) * 100
            baseline_accuracy = dev_a_baseline_metrics.get('overall_accuracy', 0) * 100
            delta_accuracy = final_accuracy - baseline_accuracy
            st.metric("Dev A Optimized Accuracy", f"{final_accuracy:.1f}%", 
                     delta=f"{delta_accuracy:+.1f}%" if delta_accuracy != 0 else None)
        else:
            st.metric("Dev A Optimized Accuracy", "N/A")
    
    with col3:
        # Iterations completed
        iterations = optimization_results.get('total_iterations', 0)
        st.metric("Iterations", iterations)
    
    with col4:
        # Duration - calculate from timestamps if available
        duration = st.session_state.get('optimization_duration', 'N/A')
        
        # If no duration from session, try to calculate from timestamps in data
        if duration == 'N/A' and data:
            try:
                # Get baseline timestamp
                baseline_timestamp = data.get('baseline_prompt', {}).get('timestamp')
                
                # Get final timestamp from optimization results
                optimization_results = data.get('optimization_results', {})
                final_metrics = optimization_results.get('final_metrics', {})
                
                # Look for the latest timestamp in the final metrics
                latest_timestamp = None
                if 'detailed_failed_cases' in final_metrics:
                    failed_cases = final_metrics['detailed_failed_cases'].get('wrong_classifications', [])
                    if failed_cases:
                        # Get the latest timestamp from failed cases
                        timestamps = [case.get('timestamp') for case in failed_cases if case.get('timestamp')]
                        if timestamps:
                            latest_timestamp = max(timestamps)
                
                # Calculate duration if both timestamps available
                if baseline_timestamp and latest_timestamp:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(baseline_timestamp.replace('Z', '+00:00') if baseline_timestamp.endswith('Z') else baseline_timestamp)
                    end_time = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00') if latest_timestamp.endswith('Z') else latest_timestamp)
                    
                    elapsed = end_time - start_time
                    total_seconds = int(elapsed.total_seconds())
                    
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    
                    if hours > 0:
                        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration = f"{minutes:02d}:{seconds:02d}"
            except Exception as e:
                pass  # Keep duration as 'N/A' if calculation fails
        
        st.metric("Duration", duration)
    
    st.divider()
    
    # === DETAILED METRICS COMPARISON ===
    st.subheader("Dev A Optimization Analysis (Primary Target)")
    
    if dev_a_baseline_metrics and dev_a_optimized_metrics:
        # Core metrics comparison using Dev A data
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dev A Metrics", "🎯 F1 Scores", "📊 Test Data Validation", "🔄 Iteration History"])
        
        with tab1:
            st.markdown("**Dev A Performance Metrics**")
            
            # Prepare comprehensive metrics data using Dev A
            metrics_data = []
            
            # Overall accuracy
            baseline_acc = dev_a_baseline_metrics.get('overall_accuracy', 0)
            optimized_acc = dev_a_optimized_metrics.get('overall_accuracy', 0)
            metrics_data.append({
                'Metric': 'Overall Accuracy',
                'Baseline': f"{baseline_acc * 100:.1f}%",
                'Optimized': f"{optimized_acc * 100:.1f}%",
                'Improvement': f"{(optimized_acc - baseline_acc) * 100:+.1f}%",
                'Raw_Baseline': baseline_acc,
                'Raw_Optimized': optimized_acc
            })
            
            # Valid JSON Rate
            baseline_json = dev_a_baseline_metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0)
            optimized_json = dev_a_optimized_metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0)
            metrics_data.append({
                'Metric': 'Valid JSON Rate',
                'Baseline': f"{baseline_json * 100:.1f}%",
                'Optimized': f"{optimized_json * 100:.1f}%", 
                'Improvement': f"{(optimized_json - baseline_json) * 100:+.1f}%",
                'Raw_Baseline': baseline_json,
                'Raw_Optimized': optimized_json
            })
            
            # Average F1 Score
            baseline_f1 = dev_a_baseline_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
            optimized_f1 = dev_a_optimized_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
            metrics_data.append({
                'Metric': 'Average F1 Score',
                'Baseline': f"{baseline_f1 * 100:.1f}%",
                'Optimized': f"{optimized_f1 * 100:.1f}%",
                'Improvement': f"{(optimized_f1 - baseline_f1) * 100:+.1f}%",
                'Raw_Baseline': baseline_f1,
                'Raw_Optimized': optimized_f1
            })
            
            # Failed cases
            baseline_failed = len(dev_a_baseline_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', []))
            optimized_failed = len(dev_a_optimized_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', []))
            metrics_data.append({
                'Metric': 'Failed Cases',
                'Baseline': str(baseline_failed),
                'Optimized': str(optimized_failed),
                'Improvement': f"{optimized_failed - baseline_failed:+d}" if baseline_failed > 0 else "0",
                'Raw_Baseline': baseline_failed,
                'Raw_Optimized': optimized_failed
            })
            
            # Create DataFrame for display
            df_metrics = pd.DataFrame(metrics_data)
            
            # Display metrics table
            st.dataframe(
                df_metrics[['Metric', 'Baseline', 'Optimized', 'Improvement']],
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization - Single comprehensive chart
            chart_data = []
            
            # Add percentage metrics (convert to percentage)
            for metric in metrics_data[:3]:  # First 3 are percentage metrics
                chart_data.extend([
                    {'Metric': metric['Metric'], 'Type': 'Baseline', 'Value': metric['Raw_Baseline'] * 100},
                    {'Metric': metric['Metric'], 'Type': 'Optimized', 'Value': metric['Raw_Optimized'] * 100}
                ])
            
            # Add failed cases metric (keep as count)
            failed_cases_metric = metrics_data[3]  # Failed cases is the 4th metric
            chart_data.extend([
                {'Metric': 'Failed Cases', 'Type': 'Baseline', 'Value': failed_cases_metric['Raw_Baseline']},
                {'Metric': 'Failed Cases', 'Type': 'Optimized', 'Value': failed_cases_metric['Raw_Optimized']}
            ])
            
            df_chart = pd.DataFrame(chart_data)
            fig1 = px.bar(
                df_chart,
                x='Metric', 
                y='Value',
                color='Type',
                title='Performance Comparison - All Metrics',
                barmode='group',
                color_discrete_sequence=['#ff7f7f', '#7fbf7f']
            )
            
            # Update layout to show different scales clearly
            fig1.update_layout(
                yaxis_title='Value (% for accuracy metrics, count for failed cases)',
                xaxis_title='Metrics',
                legend_title='Type'
            )
            
            # Add text annotations to show exact values
            fig1.update_traces(texttemplate='%{y}', textposition='outside')
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            st.markdown("**Dev A F1 Score Analysis by Field**")
            
            # Field-specific F1 scores from Dev A data
            baseline_enum_metrics = dev_a_baseline_metrics.get('enum_field_metrics', {})
            optimized_enum_metrics = dev_a_optimized_metrics.get('enum_field_metrics', {})
            
            if baseline_enum_metrics or optimized_enum_metrics:
                # Combine all fields
                all_fields = set(baseline_enum_metrics.keys()) | set(optimized_enum_metrics.keys())
                
                f1_data = []
                for field in sorted(all_fields):
                    baseline_f1 = baseline_enum_metrics.get(field, {}).get('macro_f1', 0)
                    optimized_f1 = optimized_enum_metrics.get(field, {}).get('macro_f1', 0)
                    improvement = optimized_f1 - baseline_f1
                    
                    f1_data.append({
                        'Field': field,
                        'Baseline F1': f"{baseline_f1 * 100:.1f}%",
                        'Optimized F1': f"{optimized_f1 * 100:.1f}%",
                        'Improvement': f"{improvement * 100:+.1f}%",
                        'Raw_Baseline': baseline_f1,
                        'Raw_Optimized': optimized_f1
                    })
                
                df_f1 = pd.DataFrame(f1_data)
                st.dataframe(df_f1[['Field', 'Baseline F1', 'Optimized F1', 'Improvement']], 
                           use_container_width=True, hide_index=True)
                
                # F1 Score visualization
                chart_data = []
                for _, row in df_f1.iterrows():
                    chart_data.extend([
                        {'Field': row['Field'], 'Type': 'Baseline', 'F1 Score': row['Raw_Baseline'] * 100},
                        {'Field': row['Field'], 'Type': 'Optimized', 'F1 Score': row['Raw_Optimized'] * 100}
                    ])
                
                df_f1_chart = pd.DataFrame(chart_data)
                fig_f1 = px.bar(
                    df_f1_chart,
                    x='Field',
                    y='F1 Score',
                    color='Type',
                    title='F1 Scores by Field',
                    barmode='group',
                    color_discrete_sequence=['#ff7f7f', '#7fbf7f']
                )
                fig_f1.update_layout(yaxis_title='F1 Score (%)')
                st.plotly_chart(fig_f1, use_container_width=True)
            else:
                st.info("No field-wise F1 score data available")
        
        with tab3:
            st.markdown("**Test Data Validation (Hidden Data)**")
            
            if test_metrics:
                # Show test data performance for validation
                st.info("📊 Test data results are used for hidden validation only - optimization targets Dev A metrics.")
                
                col_test1, col_test2 = st.columns(2)
                
                with col_test1:
                    # Test accuracy comparison
                    if train_baseline_metrics:
                        test_baseline_acc = train_baseline_metrics.get('overall_accuracy', 0)  # Use train as baseline reference
                        test_optimized_acc = test_metrics.get('overall_accuracy', 0)
                        test_improvement = test_optimized_acc - test_baseline_acc
                        
                        st.metric(
                            "Test Accuracy", 
                            f"{test_optimized_acc * 100:.1f}%",
                            delta=f"{test_improvement * 100:+.1f}%"
                        )
                
                with col_test2:
                    # Test F1 score
                    if train_baseline_metrics:
                        test_baseline_f1 = train_baseline_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
                        test_optimized_f1 = test_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
                        test_f1_improvement = test_optimized_f1 - test_baseline_f1
                        
                        st.metric(
                            "Test F1 Score", 
                            f"{test_optimized_f1 * 100:.1f}%",
                            delta=f"{test_f1_improvement * 100:+.1f}%"
                        )
                
                st.markdown("**Test vs Dev A Comparison**")
                
                # Create comparison table
                comparison_data = []
                if dev_a_optimized_metrics:
                    dev_a_acc = dev_a_optimized_metrics.get('overall_accuracy', 0)
                    test_acc = test_metrics.get('overall_accuracy', 0)
                    
                    dev_a_f1 = dev_a_optimized_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
                    test_f1 = test_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
                    
                    comparison_data = [
                        {"Metric": "Overall Accuracy", "Dev A": f"{dev_a_acc * 100:.1f}%", "Test": f"{test_acc * 100:.1f}%", "Difference": f"{(test_acc - dev_a_acc) * 100:+.1f}%"},
                        {"Metric": "Average F1 Score", "Dev A": f"{dev_a_f1 * 100:.1f}%", "Test": f"{test_f1 * 100:.1f}%", "Difference": f"{(test_f1 - dev_a_f1) * 100:+.1f}%"}
                    ]
                    
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            else:
                st.warning("No test data results available.")
        
        with tab4:
            st.markdown("**Dev A Error and Validation Analysis**")
            
            col_err1, col_err2 = st.columns(2)
            
            with col_err1:
                st.markdown("**Dev A Validation Metrics**")
                
                validation_data = []
                
                # JSON validity using Dev A data
                baseline_val = dev_a_baseline_metrics.get('validation_metrics', {})
                optimized_val = dev_a_optimized_metrics.get('validation_metrics', {})
                
                for metric_name, display_name in [
                    ('valid_json_accuracy', 'Valid JSON Rate'),
                    ('schema_compliance_accuracy', 'Schema Compliance'),
                    ('all_fields_present_accuracy', 'All Fields Present')
                ]:
                    baseline_val_metric = baseline_val.get(metric_name, 0)
                    optimized_val_metric = optimized_val.get(metric_name, 0)
                    
                    validation_data.append({
                        'Metric': display_name,
                        'Baseline': f"{baseline_val_metric * 100:.1f}%",
                        'Optimized': f"{optimized_val_metric * 100:.1f}%",
                        'Improvement': f"{(optimized_val_metric - baseline_val_metric) * 100:+.1f}%"
                    })
                
                if validation_data:
                    df_val = pd.DataFrame(validation_data)
                    st.dataframe(df_val, use_container_width=True, hide_index=True)
            
            with col_err2:
                st.markdown("**Dev A Failed Cases Analysis**")
                
                baseline_failed = len(dev_a_baseline_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', []))
                optimized_failed = len(dev_a_optimized_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', []))
                total_cases = dev_a_baseline_metrics.get('evaluation_metadata', {}).get('num_samples', 1)
                
                failure_data = [
                    {
                        'Stage': 'Dev A Baseline',
                        'Failed Cases': baseline_failed,
                        'Success Rate': f"{((total_cases - baseline_failed) / total_cases) * 100:.1f}%"
                    },
                    {
                        'Stage': 'Dev A Optimized', 
                        'Failed Cases': optimized_failed,
                        'Success Rate': f"{((total_cases - optimized_failed) / total_cases) * 100:.1f}%"
                    }
                ]
                
                df_failures = pd.DataFrame(failure_data)
                st.dataframe(df_failures, use_container_width=True, hide_index=True)
                
                # Failure reduction metric
                if baseline_failed > 0:
                    reduction = ((baseline_failed - optimized_failed) / baseline_failed) * 100
                    st.metric("Failure Reduction", f"{reduction:.1f}%",
                            delta=f"{optimized_failed - baseline_failed:+d} cases")
    
    elif iterations_data:  # Show iteration history even if detailed metrics are missing
        st.subheader("🔄 Iteration History")
        tab1, = st.tabs(["Iteration Progress"])
        
        with tab1:
            st.markdown("**Dev A Iteration-by-Iteration Progress**")
            
            # Process iteration data
            iteration_progress = []
            
            selected_iterations = [iter_data for iter_data in iterations_data if iter_data.get('type') == 'selected']
            selected_iterations.sort(key=lambda x: x.get('iteration', 0))
            
            for iter_data in selected_iterations:
                if 'selected_prompt' in iter_data:
                    selected = iter_data['selected_prompt']
                    iteration_progress.append({
                        'Iteration': iter_data.get('iteration', 0),
                        'Accuracy': f"{selected.get('dev_a_metrics', {}).get('overall_accuracy', 0) * 100:.1f}%",
                        'F1 Score': f"{selected.get('dev_a_metrics', {}).get('summary', {}).get('average_enum_macro_f1', 0) * 100:.1f}%",
                        'Valid JSON': f"{selected.get('dev_a_metrics', {}).get('validation_metrics', {}).get('valid_json_accuracy', 0) * 100:.1f}%",
                        'Improvement': f"{selected.get('improvement_over_baseline', 0) * 100:+.1f}%"
                    })
            
            if iteration_progress:
                df_iter = pd.DataFrame(iteration_progress)
                st.dataframe(df_iter, use_container_width=True, hide_index=True)
                
                # Iteration progress chart
                chart_data = []
                for i, iter_data in enumerate(selected_iterations):
                    if 'selected_prompt' in iter_data:
                        selected = iter_data['selected_prompt']
                        metrics = selected.get('dev_a_metrics', {})
                        
                        chart_data.append({
                            'Iteration': iter_data.get('iteration', 0),
                            'Accuracy': metrics.get('overall_accuracy', 0) * 100,
                            'F1 Score': metrics.get('summary', {}).get('average_enum_macro_f1', 0) * 100,
                            'Valid JSON': metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0) * 100
                        })
                
                if chart_data:
                    df_progress = pd.DataFrame(chart_data)
                    fig_progress = px.line(
                        df_progress,
                        x='Iteration',
                        y=['Accuracy', 'F1 Score', 'Valid JSON'],
                        title='Dev A Progress Across Iterations',
                        markers=True
                    )
                    fig_progress.update_layout(yaxis_title='Percentage (%)')
                    st.plotly_chart(fig_progress, use_container_width=True)
            else:
                st.info("No iteration progress data available")
    
    else:
        st.warning("Insufficient data for detailed analysis. Some metrics may be missing.")
    
    st.divider()
    
    # === OPTIMIZATION INSIGHTS ===
    st.subheader("🔍 Dev A Optimization Insights")
    
    col_insights1, col_insights2 = st.columns(2)
    
    with col_insights1:
        st.markdown("**Key Dev A Improvements**")
        insights = []
        
        if dev_a_baseline_metrics and dev_a_optimized_metrics:
            # Accuracy improvement on Dev A
            acc_improvement = (dev_a_optimized_metrics.get('overall_accuracy', 0) - dev_a_baseline_metrics.get('overall_accuracy', 0)) * 100
            if acc_improvement > 0:
                insights.append(f"✅ **Dev A Accuracy increased by {acc_improvement:.1f}%**")
            elif acc_improvement < 0:
                insights.append(f"⚠️ **Dev A Accuracy decreased by {abs(acc_improvement):.1f}%**")
            
            # F1 improvement on Dev A
            baseline_f1 = dev_a_baseline_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
            optimized_f1 = dev_a_optimized_metrics.get('summary', {}).get('average_enum_macro_f1', 0)
            f1_improvement = (optimized_f1 - baseline_f1) * 100
            if f1_improvement > 0:
                insights.append(f"✅ **Dev A F1 Score improved by {f1_improvement:.1f}%**")
            elif f1_improvement < 0:
                insights.append(f"⚠️ **Dev A F1 Score decreased by {abs(f1_improvement):.1f}%**")
            
            # Failed cases improvement on Dev A
            baseline_failed = len(dev_a_baseline_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', []))
            optimized_failed = len(dev_a_optimized_metrics.get('detailed_failed_cases', {}).get('wrong_classifications', []))
            if baseline_failed > optimized_failed:
                insights.append(f"✅ **Reduced Dev A failed cases by {baseline_failed - optimized_failed}**")
            elif optimized_failed > baseline_failed:
                insights.append(f"⚠️ **Dev A failed cases increased by {optimized_failed - baseline_failed}**")
            
            # JSON validity on Dev A
            json_improvement = (optimized_json - baseline_json) * 100
            if json_improvement > 0:
                insights.append(f"✅ **Dev A JSON validity improved by {json_improvement:.1f}%**")
            elif json_improvement < 0:
                insights.append(f"⚠️ **Dev A JSON validity decreased by {abs(json_improvement):.1f}%**")
        
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("No significant improvements detected")
    
    with col_insights2:
        st.markdown("**Optimization Summary**")
        
        # Stopping reason with explanations
        stopping_reason = optimization_results.get('stopping_reason', 'Unknown')
        
        # Create user-friendly explanations
        reason_explanations = {
            'adaptive_stopping': '🎯 Adaptive stopping - optimal point detected',
            'converged': '✅ Converged - improvements stabilized with good performance',
            'poor_performance_convergence': '⚠️ Converged with poor performance',
            'max_iterations_reached': '🔄 Maximum iterations reached',
            'below_threshold': '⏹️ Improvement below threshold',
            'max_retries_reached': '⚠️ Maximum retry attempts reached',
            'Unknown': '❓ Unknown stopping condition'
        }
        
        reason_display = reason_explanations.get(stopping_reason, f"❓ {stopping_reason}")
        st.markdown(f"**Stopping Reason:** {reason_display}")
        
        # Total iterations vs max
        total_iterations = optimization_results.get('total_iterations', 0)
        st.markdown(f"**Iterations Used:** {total_iterations}")
        
        # Best iteration
        best_candidate = optimization_results.get('best_candidate', {})
        if best_candidate:
            best_improvement = best_candidate.get('improvement_over_baseline', 0) * 100
            st.markdown(f"**Best Improvement:** {best_improvement:+.1f}%")
        
        # Human feedback insights
        human_feedback = optimization_results.get('human_feedback_summary', {})
        if human_feedback and isinstance(human_feedback, dict):
            cases_reviewed = human_feedback.get('total_cases_reviewed', 0)
            if cases_reviewed > 0:
                st.markdown(f"**Human Feedback:** {cases_reviewed} cases reviewed")
    
    st.divider()
    
    # === PROMPTS COMPARISON ===
    st.subheader("📝 Prompt Analysis")
    
    col_prompts1, col_prompts2 = st.columns(2)
    
    with col_prompts1:
        st.markdown("**Baseline Prompt**")
        baseline_prompt = data.get('baseline_prompt', 'No baseline prompt available').get("enhanced_system_prompt")
        st.code(baseline_prompt, language="text", line_numbers=False)
    
    with col_prompts2:
        st.markdown("**Optimized Prompt**")
        final_prompt = final_results.get('recommended_prompt', 'No optimized prompt available')
        st.code(final_prompt, language="text", line_numbers=False)
    
    # === RAW DATA INSPECTION ===
    with st.expander("🔧 Raw Data Inspection", expanded=False):
        st.markdown("**Available Data Keys:**")
        st.json(list(data.keys()))
        
        st.markdown("**Sample of Raw Backend Response:**")
        # Show a subset of the raw data for debugging
        sample_data = {
            'final_results_keys': list(final_results.keys()) if final_results else [],
            'train_baseline_metrics_keys': list(train_baseline_metrics.keys()) if train_baseline_metrics else [],
            'dev_a_baseline_metrics_keys': list(dev_a_baseline_metrics.keys()) if dev_a_baseline_metrics else [],
            'dev_a_optimized_metrics_keys': list(dev_a_optimized_metrics.keys()) if dev_a_optimized_metrics else [],
            'test_metrics_keys': list(test_metrics.keys()) if test_metrics else [],
            'optimization_results_keys': list(optimization_results.keys()) if optimization_results else [],
            'iterations_count': len(iterations_data)
        }
        st.json(sample_data)

# Past Reports
def show_past_reports():
    st.header("Past Optimization Reports")
    
    # If we just completed an optimization, show its results first
    if st.session_state.optimization_results:
        st.subheader("🎉 Latest Optimization Results")
        show_results()
        st.divider()
        st.subheader("📝 All Reports")
    
    # Fetch all optimizations
    with st.spinner("Loading past reports..."):
        optimizations_data, error = make_api_call("/optimizations")
    
    if error:
        st.error(f"Failed to load reports: {error}")
        return
    
    optimizations = optimizations_data.get('optimizations', [])
    
    if not optimizations:
        st.info("No past optimizations found.")
        return
    
    # Summary stats
    completed = [opt for opt in optimizations if opt['status'] == 'completed']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Optimizations", len(optimizations))
    with col2:
        st.metric("Completed", len(completed))
    with col3:
        if completed:
            # Fix average improvement calculation - handle decimal vs percentage format
            improvements = []
            for opt in completed:
                improvement = opt.get('improvement', 0)
                if improvement > 0:
                    # Convert to percentage if it's in decimal format
                    if improvement <= 1:
                        improvement = improvement * 100
                    improvements.append(improvement)
            
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
            else:
                st.metric("Avg Improvement", "N/A")
        else:
            st.metric("Avg Improvement", "N/A")
    
    # List optimizations
    st.subheader("Reports History")
    
    for i, opt in enumerate(optimizations):
        with st.container():
            col_info, col_metrics, col_action = st.columns([3, 2, 1])
            
            with col_info:
                status_colors = {
                    'completed': '🟢',
                    'optimization_completed': '🟡',
                    'in_progress': '🔵',
                    'initialized': '🟠'
                }
                status_icon = status_colors.get(opt['status'], '⚪')
                
                st.markdown(f"**{status_icon} Optimization {i+1}**")
                # Make ID more readable by showing full ID in smaller font
                st.markdown(f"<small><code>{opt['request_id']}</code></small>", unsafe_allow_html=True)
                
                try:
                    created = datetime.fromisoformat(opt['created'].replace('Z', '+00:00'))
                    st.caption(f"Created: {created.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.caption(f"Created: {opt.get('created', 'Unknown')[:16]}")
            
            with col_metrics:
                # Fix improvement calculation - multiply by 100 to convert from decimal to percentage
                improvement_value = opt.get('improvement', 0)
                if improvement_value > 0:
                    # If the value is already > 1, it's probably already in percentage format
                    if improvement_value > 1:
                        st.metric("Improvement", f"{improvement_value:.1f}%")
                    else:
                        # Convert decimal to percentage
                        st.metric("Improvement", f"{improvement_value * 100:.1f}%")
                else:
                    st.metric("Improvement", "N/A")
                st.caption(f"Iterations: {opt.get('iterations', 0)}")
            
            with col_action:
                if st.button("View Report", key=f"view_{opt['request_id']}"):
                    with st.spinner("Loading report..."):
                        results, load_error = make_api_call(f"/optimize/{opt['request_id']}/results")
                        
                        if load_error:
                            st.error(f"Failed to load: {load_error}")
                        else:
                            st.session_state.optimization_results = results
                            st.session_state.current_step = 'results'
                            st.rerun()
            
            st.divider()

# Main app
def main():
    init_session_state()
    show_header()
    show_navigation()
    
    # Auto-check optimization status if running
    if st.session_state.optimization_running:
        completed = check_optimization_status()
        if completed:
            st.rerun()
    
    # Route to current step
    if st.session_state.current_step == 'upload':
        show_upload_step()
    elif st.session_state.current_step == 'config':
        show_config_step()
    elif st.session_state.current_step == 'results':
        show_results()
    elif st.session_state.current_step == 'reports':
        show_past_reports()

if __name__ == "__main__":
    main()