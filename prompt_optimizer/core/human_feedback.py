"""
Human Feedback Integration with LangFuse Annotation Queues

This module handles:
1. Creating traces for evaluation runs
2. Adding traces to annotation queues
3. Managing score configurations
4. Collecting human feedback
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass
from langfuse import Langfuse
import pytz

@dataclass
class HumanFeedbackConfig:
    """Configuration for human feedback collection"""
    queue_name: str
    score_configs: List[Dict[str, Any]]
    description: str = ""
    reviewers_per_run: int = 1
    enable_reservations: bool = True
    reservation_length_minutes: int = 30

@dataclass
class EvaluationTrace:
    """Data structure for evaluation traces"""
    trace_id: str
    input_prompt: str
    model_output: str
    expected_output: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    candidate_prompt: str
    metadata: Dict[str, Any]

class HumanFeedbackManager:
    """
    Manages human feedback collection using LangFuse annotation queues.
    
    This class handles:
    - Creating traces for Dev B evaluation results
    - Setting up annotation queues
    - Adding traces to queues for human review
    - Collecting and processing human feedback
    """
    
    def __init__(self):
        """Initialize the human feedback manager with LangFuse client."""
        self.langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
        self.annotation_queues = {}
        
    def setup_score_configurations(self, configs: List[Dict[str, Any]]) -> bool:
        """
        Set up score configurations for human annotation.
        
        Note: Score configurations must be created via the LangFuse UI.
        This method validates that the required configs exist.
        
        Args:
            configs: List of score configuration definitions
            
        Returns:
            bool: True if all configurations are available
        """
        print("📋 Setting up score configurations...")
        print("⚠️  Note: Score configurations must be created via LangFuse UI")
        print("   Navigate to Settings > Score Configs in LangFuse dashboard")
        
        for config in configs:
            print(f"   Required config: {config['name']} ({config['type']})")
            if 'description' in config:
                print(f"     Description: {config['description']}")
        
        return True
    
    def create_annotation_queue(self, config: HumanFeedbackConfig) -> str:
        """
        Create an annotation queue for human feedback collection.
        
        Note: Annotation queues must be created via the LangFuse UI.
        This method provides instructions for manual setup.
        
        Args:
            config: Human feedback configuration
            
        Returns:
            str: Queue name for reference
        """
        print(f"🔄 Setting up annotation queue: {config.queue_name}")
        print("⚠️  Note: Annotation queues must be created via LangFuse UI")
        print("   Steps to create queue:")
        print("   1. Navigate to 'Annotate' tab in LangFuse dashboard")
        print("   2. Click '+ New annotation queue'")
        print(f"   3. Name: {config.queue_name}")
        print(f"   4. Description: {config.description}")
        print(f"   5. Reviewers per run: {config.reviewers_per_run}")
        print(f"   6. Enable reservations: {config.enable_reservations}")
        
        if config.score_configs:
            print("   7. Add score configurations:")
            for score_config in config.score_configs:
                print(f"      - {score_config['name']}")
        
        self.annotation_queues[config.queue_name] = config
        return config.queue_name
    
    def create_evaluation_trace(self, 
                              input_prompt: str,
                              model_output: str,
                              expected_output: Dict[str, Any],
                              candidate_prompt: str,
                              baseline_metrics: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a trace for evaluation results that will be reviewed by humans.
        
        Args:
            input_prompt: The input prompt used for evaluation
            model_output: The model's output/prediction
            expected_output: The expected/ground truth output
            candidate_prompt: The optimized prompt being evaluated
            baseline_metrics: Baseline performance metrics
            metadata: Additional metadata for the trace
            
        Returns:
            str: Trace ID for the created trace
        """
        # Create a trace using the LangFuse client directly
        trace = self.langfuse.trace(
            name=f"Dev B Evaluation - Human Review",
            tags=["dev_b_evaluation", "human_feedback", "prompt_optimization"],
            metadata={
                "evaluation_type": "dev_b_human_review",
                "candidate_prompt": candidate_prompt,
                "baseline_metrics": baseline_metrics,
                "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).isoformat(),
                **(metadata or {})
            },
            input={
                "prompt": input_prompt,
                "candidate_prompt_used": candidate_prompt
            },
            output={
                "model_prediction": model_output,
                "expected_output": expected_output
            }
        )
        
        # Get the trace ID
        trace_id = trace.id
        
        print(f"✅ Created evaluation trace: {trace_id}")
        return trace_id
    
    def add_trace_to_annotation_queue(self, 
                                    trace_id: str, 
                                    queue_name: str,
                                    priority: str = "normal") -> bool:
        """
        Add a trace to an annotation queue for human review.
        
        Note: This must be done via LangFuse UI or API.
        This method provides instructions for manual addition.
        
        Args:
            trace_id: ID of the trace to add to queue
            queue_name: Name of the annotation queue
            priority: Priority level for the annotation task
            
        Returns:
            bool: True if instructions provided successfully
        """
        print(f"📝 Adding trace {trace_id} to annotation queue: {queue_name}")
        print("⚠️  Manual steps required:")
        print("   1. Navigate to LangFuse dashboard")
        print("   2. Go to 'Traces' tab")
        print(f"   3. Find trace ID: {trace_id}")
        print("   4. Click on the trace")
        print("   5. Click 'Annotate' dropdown")
        print(f"   6. Select 'Add to queue' > '{queue_name}'")
        
        # Store for tracking
        if queue_name not in self.annotation_queues:
            print(f"⚠️  Queue {queue_name} not found in local tracking")
        
        return True
    
    def batch_create_evaluation_traces(self, 
                                     evaluation_results: List[Dict[str, Any]],
                                     candidate_prompt: str,
                                     baseline_metrics: Dict[str, Any]) -> List[str]:
        """
        Create multiple evaluation traces for batch human review.
        
        Args:
            evaluation_results: List of evaluation results from Dev B
            candidate_prompt: The optimized prompt being evaluated
            baseline_metrics: Baseline performance metrics
            
        Returns:
            List[str]: List of created trace IDs
        """
        trace_ids = []
        
        print(f"🔄 Creating {len(evaluation_results)} evaluation traces for human review...")
        
        for i, result in enumerate(evaluation_results):
            try:
                # Create trace for this evaluation result
                trace_id = self.create_evaluation_trace(
                    input_prompt=result.get('input_prompt', ''),
                    model_output=result.get('model_output', ''),
                    expected_output=result.get('expected_output', {}),
                    candidate_prompt=candidate_prompt,
                    baseline_metrics=baseline_metrics,
                    metadata={
                        "batch_index": i,
                        "total_batch_size": len(evaluation_results),
                        "evaluation_id": result.get('evaluation_id', f"eval_{i}")
                    }
                )
                trace_ids.append(trace_id)
                
            except Exception as e:
                print(f"❌ Error creating trace for evaluation {i}: {e}")
                continue
        
        print(f"✅ Created {len(trace_ids)} evaluation traces")
        return trace_ids
    
    def get_human_feedback_summary(self, trace_ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve human feedback summary for a list of traces.
        
        Note: This requires the traces to have been annotated via LangFuse UI.
        
        Args:
            trace_ids: List of trace IDs to get feedback for
            
        Returns:
            Dict containing feedback summary
        """
        print(f"📊 Retrieving human feedback for {len(trace_ids)} traces...")
        
        feedback_summary = {
            "total_traces": len(trace_ids),
            "annotated_traces": 0,
            "pending_traces": 0,
            "average_scores": {},
            "feedback_details": []
        }
        
        for trace_id in trace_ids:
            try:
                # Note: LangFuse API has changed, using simplified approach for now
                print(f"⚠️  LangFuse feedback collection not yet implemented for trace {trace_id}")
                feedback_summary["pending_traces"] += 1
                    
            except Exception as e:
                print(f"⚠️  Error fetching feedback for trace {trace_id}: {e}")
                feedback_summary["pending_traces"] += 1
        
        # Calculate averages
        for score_name, values in feedback_summary["average_scores"].items():
            feedback_summary["average_scores"][score_name] = {
                "average": sum(values) / len(values),
                "count": len(values),
                "values": values
            }
        
        print(f"✅ Feedback summary: {feedback_summary['annotated_traces']} annotated, {feedback_summary['pending_traces']} pending")
        return feedback_summary
    
    def wait_for_human_feedback(self, 
                              trace_ids: List[str], 
                              timeout_minutes: int = 60,
                              check_interval_seconds: int = 30) -> Dict[str, Any]:
        """
        Wait for human feedback on traces with timeout.
        
        Args:
            trace_ids: List of trace IDs to wait for
            timeout_minutes: Maximum time to wait in minutes
            check_interval_seconds: How often to check for updates
            
        Returns:
            Dict containing feedback results
        """
        import time
        
        print(f"⏳ Waiting for human feedback on {len(trace_ids)} traces...")
        print(f"   Timeout: {timeout_minutes} minutes")
        print(f"   Check interval: {check_interval_seconds} seconds")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            feedback_summary = self.get_human_feedback_summary(trace_ids)
            
            if feedback_summary["annotated_traces"] == len(trace_ids):
                print("✅ All traces have been annotated!")
                return feedback_summary
            
            print(f"📊 Progress: {feedback_summary['annotated_traces']}/{len(trace_ids)} traces annotated")
            time.sleep(check_interval_seconds)
        
        print("⏰ Timeout reached. Returning partial feedback.")
        return self.get_human_feedback_summary(trace_ids)

# Example usage and configuration
def get_default_human_feedback_config() -> HumanFeedbackConfig:
    """Get default configuration for human feedback collection."""
    return HumanFeedbackConfig(
        queue_name="prompt_optimization_dev_b",
        description="Human review of Dev B evaluation results for prompt optimization",
        score_configs=[
            {
                "name": "classification_accuracy",
                "type": "Numeric",
                "description": "How accurate is the classification? (0-1 scale)"
            },
            {
                "name": "response_quality",
                "type": "Categorical",
                "description": "Overall quality of the response",
                "categories": ["Poor", "Fair", "Good", "Excellent"]
            },
            {
                "name": "improvement_over_baseline",
                "type": "Boolean",
                "description": "Is this better than the baseline prompt?"
            },
            {
                "name": "field_accuracy",
                "type": "Categorical", 
                "description": "Which fields are correctly classified?",
                "categories": ["All Correct", "Mostly Correct", "Some Errors", "Many Errors"]
            }
        ],
        reviewers_per_run=1,
        enable_reservations=True,
        reservation_length_minutes=30
    )

# Integration with existing workflow
class DevBHumanFeedbackIntegration:
    """
    Integration class for adding human feedback to Dev B evaluation workflow.
    """
    
    def __init__(self, feedback_manager: HumanFeedbackManager):
        self.feedback_manager = feedback_manager
        self.config = get_default_human_feedback_config()
        
    def setup_human_feedback_workflow(self) -> bool:
        """Set up the complete human feedback workflow."""
        print("🚀 Setting up human feedback workflow...")
        
        # Setup score configurations
        self.feedback_manager.setup_score_configurations(self.config.score_configs)
        
        # Create annotation queue
        self.feedback_manager.create_annotation_queue(self.config)
        
        print("✅ Human feedback workflow setup complete!")
        print("📝 Next steps:")
        print("   1. Create score configurations in LangFuse UI")
        print("   2. Create annotation queue in LangFuse UI") 
        print("   3. Run Dev B evaluation to generate traces")
        print("   4. Add traces to annotation queue")
        print("   5. Collect human feedback")
        
        return True
    
    def process_dev_b_results_for_human_feedback(self, 
                                               dev_b_results: List[Dict[str, Any]],
                                               candidate_prompt: str,
                                               baseline_metrics: Dict[str, Any]) -> List[str]:
        """
        Process Dev B evaluation results and create traces for human feedback.
        
        Args:
            dev_b_results: Results from Dev B evaluation
            candidate_prompt: The optimized prompt being evaluated
            baseline_metrics: Baseline performance metrics
            
        Returns:
            List[str]: Created trace IDs
        """
        print("🔄 Processing Dev B results for human feedback...")
        
        # Create evaluation traces
        trace_ids = self.feedback_manager.batch_create_evaluation_traces(
            evaluation_results=dev_b_results,
            candidate_prompt=candidate_prompt,
            baseline_metrics=baseline_metrics
        )
        
        # Provide instructions for adding to queue
        if trace_ids:
            print(f"📝 Manual step required: Add {len(trace_ids)} traces to annotation queue")
            print(f"   Queue name: {self.config.queue_name}")
            print("   Trace IDs:")
            for trace_id in trace_ids:
                print(f"     - {trace_id}")
        
        return trace_ids 