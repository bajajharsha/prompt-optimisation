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

def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>Prompt Optimization Dashboard</h1>
        <p>Optimize your prompts with automated testing and human feedback</p>
    </div>
    """, unsafe_allow_html=True)

def show_navigation():
    """Simple horizontal navigation"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("1. Upload Dataset", key="nav_upload"):
            st.session_state.current_step = 'upload'
            st.rerun()
    
    with col2:
        if st.button("2. Configuration", key="nav_config"):
            st.session_state.current_step = 'config'
            st.rerun()
    
    with col3:
        if st.button("3. Optimization", key="nav_optimize"):
            st.session_state.current_step = 'optimize'
            st.rerun()
    
    with col4:
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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset selection
        st.subheader("Dataset Selection")
        
        # Option to use uploaded dataset or select from available datasets
        dataset_option = st.radio(
            "Choose dataset source:",
            ["Use uploaded dataset", "Select from available datasets"],
            index=0 if st.session_state.dataset_uploaded else 1
        )
        
        selected_dataset = None
        
        if dataset_option == "Use uploaded dataset":
            if st.session_state.dataset_uploaded:
                selected_dataset = st.session_state.selected_dataset
                st.success(f"Using uploaded dataset: {selected_dataset}")
            else:
                st.warning("No dataset uploaded. Please upload a dataset first or select from available datasets.")
        else:
            # Fetch available datasets
            with st.spinner("Loading available datasets..."):
                datasets_data, error = make_api_call("/datasets")
                
                if error:
                    st.error(f"Failed to load datasets: {error}")
                    st.info("You can upload a dataset in Step 1 instead.")
                else:
                    datasets = datasets_data.get('datasets', [])
                    if datasets:
                        dataset_names = [d.get('name', 'Unknown') for d in datasets]
                        selected_dataset = st.selectbox(
                            "Select a dataset:",
                            options=dataset_names,
                            help="Choose from previously uploaded datasets"
                        )
                        
                        # Show dataset info
                        if selected_dataset:
                            selected_dataset_info = next((d for d in datasets if d.get('name') == selected_dataset), None)
                            if selected_dataset_info:
                                with st.expander("Dataset Information"):
                                    st.write(f"**Name:** {selected_dataset_info.get('name', 'N/A')}")
                                    if selected_dataset_info.get('description'):
                                        st.write(f"**Description:** {selected_dataset_info.get('description')}")
                                    st.write(f"**Created:** {selected_dataset_info.get('createdAt', 'N/A')[:10]}")
                    else:
                        st.warning("No datasets available. Please upload a dataset first.")
        
        # Only proceed if we have a selected dataset
        if selected_dataset:
            # Schema configuration - simple
            st.subheader("Schema Definition")
            
            # Quick examples
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                if st.button("Load Code Generation Schema", type="secondary"):
                    st.session_state.schema_fields = {
                        "action": "CODE_GENERATION, NOT_FOUND",
                        "subAction": "CODING, VISUAL_EDITS, ERROR, GENERAL",
                        "platform": "DYNAMIC_WEB_APPLICATION, STATIC_WEB_APPLICATION, DYNAMIC_MOBILE_APP, STATIC_MOBILE_APP, NOT_FOUND",
                        "framework": "REACT, FLUTTER, NOT_FOUND",
                        "languageType": "REACT_JAVASCRIPT, NOT_FOUND"
                    }
                    st.rerun()
            with col_ex2:
                if st.button("Load Education Schema", type="secondary"):
                    st.session_state.schema_fields = {
                        "intent": "CONCEPT_EXPLANATION, PROBLEM_SOLVING, MCQ_PRACTICE, THEORY_REVIEW, REAL_WORLD_APPLICATION, EXAM_PREPARATION, NOT_FOUND",
                        "subject": "MATH, PHYSICS, CHEMISTRY, BIOLOGY, HISTORY, GEOGRAPHY, ENGLISH, COMPUTER_SCIENCE, NOT_FOUND",
                        "difficulty": "EASY, MEDIUM, HARD, NOT_FOUND",
                        "gradeLevel": "GRADE_6, GRADE_7, GRADE_8, GRADE_9, GRADE_10, GRADE_11, GRADE_12, NOT_FOUND"
                    }
                    st.rerun()
            
            # Schema input
            schema = {}
            if st.session_state.schema_fields:
                for field_name, field_values in st.session_state.schema_fields.items():
                    col_field, col_values = st.columns([1, 3])
                    with col_field:
                        st.write(f"**{field_name}**")
                    with col_values:
                        values = st.text_input(
                            f"Values:",
                            value=field_values,
                            key=f"field_{field_name}",
                            label_visibility="collapsed"
                        )
                        st.session_state.schema_fields[field_name] = values
                        if values.strip():
                            schema[field_name] = [v.strip() for v in values.split(',') if v.strip()]
                
                if st.button("Clear Schema", type="secondary"):
                    st.session_state.schema_fields = {}
                    st.rerun()
            else:
                st.info("👆 Choose an example schema above to get started")
            
            # Prompts
            st.subheader("Prompts")
            system_prompt = st.text_area(
                "System Prompt",
                value="You are a text classifier. Classify the input according to the schema and return valid JSON.",
                height=100
            )
            
            user_prompt = st.text_input(
                "User Prompt Template",
                value="Classify this text:"
            )
            
            # Model selection
            st.subheader("Model Configuration")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                provider = st.selectbox("Provider", ["groq", "anthropic", "openai", "google"])
            
            with col_m2:
                # Dynamic model options based on provider
                model_options = {
                    "groq": ["llama-3.3-70b-versatile"],
                    "anthropic": ["claude-sonnet-4-20250514"], 
                    "openai": ["gpt-4.1-mini-2025-04-14"],
                    "google": ["gemini-2.5-pro-preview-06-05"]
                }
                model_name = st.selectbox("Model", options=model_options[provider])
            
            # Start optimization button
            if st.button("Start Optimization", type="primary"):
                if schema and system_prompt:
                    if schema:
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
                            else:
                                st.session_state.request_id = result.get('request_id')
                                st.session_state.optimization_running = True
                                st.session_state.current_step = 'optimize'
                                st.success("Optimization started!")
                                st.rerun()
                    else:
                        st.error("Invalid schema format")
                elif not schema:
                    st.error("Please add at least one schema field with values")
                else:
                    st.error("Please fill in all required fields")
    
    with col2:
        st.subheader("Schema Preview")
        
        if 'schema' in locals() and schema:
            st.json(schema)
        else:
            st.info("Schema will appear here")
            st.markdown("**How to use:**")
            st.markdown("1. Click an example button")
            st.markdown("2. Edit the values if needed")
            st.markdown("3. Values should be comma-separated")

# Step 3: Optimization Progress
def check_optimization_status():
    """Check optimization status"""
    if st.session_state.optimization_running and st.session_state.request_id:
        status_data, error = make_api_call(f"/optimize/{st.session_state.request_id}/status")
        
        if error:
            # Try loading from file
            results_data, results_error = make_api_call(f"/optimize/{st.session_state.request_id}/results")
            if not results_error:
                st.session_state.optimization_running = False
                st.session_state.optimization_results = results_data
                st.session_state.current_step = 'results'
                return True
            return False
        
        progress = status_data.get('progress_percentage', 0)
        
        if progress >= 100 or status_data.get('status') == 'completed':
            # Load final results
            results_data, results_error = make_api_call(f"/optimize/{st.session_state.request_id}/results")
            if not results_error:
                st.session_state.optimization_running = False
                st.session_state.optimization_results = results_data
                st.session_state.current_step = 'results'
                return True
        
        return False
    return None

def show_optimization_step():
    st.header("Step 3: Optimization in Progress")
    
    if not st.session_state.optimization_running:
        st.warning("No optimization running. Please start from configuration step.")
        return
    
    # Check status
    completed = check_optimization_status()
    
    if completed:
        st.balloons()
        st.success("Optimization completed!")
        st.rerun()
        return
    
    # Get current status
    status_data, error = make_api_call(f"/optimize/{st.session_state.request_id}/status")
    
    if not error:
        progress = status_data.get('progress_percentage', 0)
        
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        # Progress bar
        st.progress(progress / 100)
        
        # Status info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Progress", f"{progress:.0f}%")
        
        with col2:
            current_iter = status_data.get('current_iteration')
            total_iter = status_data.get('total_iterations')
            if current_iter and total_iter:
                st.metric("Iteration", f"{current_iter}/{total_iter}")
            else:
                st.metric("Status", "Processing")
        
        with col3:
            eta = status_data.get('estimated_time_remaining')
            if eta:
                st.metric("ETA", f"{eta}s")
            else:
                st.metric("ETA", "Calculating...")
        
        # Current step
        current_step = status_data.get('current_step', 'Processing...')
        st.info(f"Current Step: {current_step}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-refresh every 3 seconds
        time.sleep(3)
        st.rerun()

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
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    elif st.session_state.current_step == 'optimize':
        show_optimization_step()
    elif st.session_state.current_step == 'results':
        show_results()
    elif st.session_state.current_step == 'reports':
        show_past_reports()

if __name__ == "__main__":
    main()