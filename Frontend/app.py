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
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=300)
        
        if response.status_code in [200, 201]:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
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
                            # Inline editing - values are directly editable
                            updated_values = st.text_input(
                                "Values:",
                                value=field_values,
                                key=f"edit_{field_name}",
                                label_visibility="collapsed"
                            )
                            # Auto-update when changed
                            if updated_values != field_values:
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
                        st.session_state.schema_fields ={
                            "intent": "CONCEPT_EXPLANATION, PROBLEM_SOLVING, MCQ_PRACTICE, THEORY_REVIEW, REAL_WORLD_APPLICATION, EXAM_PREPARATION",
                            "subject": "MATH, PHYSICS, CHEMISTRY, BIOLOGY, HISTORY, GEOGRAPHY, ENGLISH, COMPUTER_SCIENCE",
                        }
                        st.rerun()
            
            # Generate schema for optimization
            schema = {}
            if st.session_state.get('schema_fields'):
                for field_name, field_values in st.session_state.schema_fields.items():
                    if field_values.strip():
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
                    value="You are a text classifier. Classify the input according to the schema and return valid JSON.",
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
    st.header("Optimization Results")
    
    if not st.session_state.optimization_results:
        st.warning("No optimization results available.")
        return
    
    # Load from optimized_results.json structure
    data = st.session_state.optimization_results.get('data', {})
    
    # Key metrics
    st.subheader("Performance Summary")
    
    final_results = data.get('final_results', {})
    baseline_metrics = data.get('train_baseline_metrics', {})
    test_metrics = data.get('test_metrics', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        improvement = final_results.get('test_improvement', 0) * 100
        st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    
    with col2:
        if test_metrics:
            accuracy = test_metrics.get('overall_accuracy', 0)
            st.metric("Final Accuracy", f"{accuracy:.3f}")
        else:
            st.metric("Final Accuracy", "N/A")
    
    with col3:
        opt_results = data.get('optimization_results', {})
        iterations = opt_results.get('total_iterations', 0)
        st.metric("Iterations", iterations)
    
    with col4:
        duration = st.session_state.get('optimization_duration', 'N/A')
        st.metric("⏱️ Duration", duration)
    
    with col5:
        decision = final_results.get('deployment_decision', 'unknown')
        color = "🟢" if decision == "deploy" else "🟡"
        st.metric("Recommendation", f"{color} {decision.title()}")
    
    # Comparison chart
    if baseline_metrics and test_metrics:
        st.subheader("Baseline vs Optimized Comparison")
        
        comparison_data = {
            'Metric': ['Overall Accuracy', 'Valid JSON Rate'],
            'Baseline': [
                baseline_metrics.get('overall_accuracy', 0),
                baseline_metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0)
            ],
            'Optimized': [
                test_metrics.get('overall_accuracy', 0),
                test_metrics.get('validation_metrics', {}).get('valid_json_accuracy', 0)
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        df['Improvement'] = df['Optimized'] - df['Baseline']
        
        # Chart
        fig = px.bar(
            df, 
            x='Metric', 
            y=['Baseline', 'Optimized'],
            title='Performance Comparison',
            barmode='group',
            color_discrete_sequence=['#ff7f7f', '#7fbf7f']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(df.style.format({
            'Baseline': '{:.3f}',
            'Optimized': '{:.3f}',
            'Improvement': '{:+.3f}'
        }), use_container_width=True)
    
    # Final prompt
    st.subheader("Optimized Prompt")
    final_prompt = final_results.get('recommended_prompt', 'No prompt available')
    st.markdown(f"**Final Prompt:**")
    st.code(final_prompt, language="text")

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
            avg_improvement = sum(opt.get('improvement', 0) for opt in completed) / len(completed)
            st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
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
                st.caption(f"ID: {opt['request_id'][:12]}...")
                
                try:
                    created = datetime.fromisoformat(opt['created'].replace('Z', '+00:00'))
                    st.caption(f"Created: {created.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.caption(f"Created: {opt.get('created', 'Unknown')[:16]}")
            
            with col_metrics:
                if opt.get('improvement', 0) > 0:
                    st.metric("Improvement", f"{opt['improvement']:.1f}%")
                else:
                    st.metric("Improvement", "N/A")
                st.caption(f"Iterations: {opt.get('iterations', 0)}")
            
            with col_action:
                if st.button("View Report", key=f"view_{opt['request_id'][:8]}"):
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