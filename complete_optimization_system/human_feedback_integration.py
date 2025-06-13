"""
Human Feedback Integration - Simplified integration for the complete system
"""

import sys
import os
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.core.human_feedback import HumanFeedbackManager, DevBHumanFeedbackIntegration


class HumanFeedbackIntegration:
    """
    Simplified human feedback integration for the complete system
    """
    
    def __init__(self):
        self.feedback_manager = HumanFeedbackManager()
        self.feedback_integration = DevBHumanFeedbackIntegration(self.feedback_manager)
        self.setup_completed = False
    
    async def collect_feedback(
        self,
        candidate_prompt: str,
        dev_b_results: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect human feedback on Dev B results
        
        Args:
            candidate_prompt: The optimized prompt being evaluated
            dev_b_results: Results from Dev B evaluation
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Human feedback results
        """
        print("👥 Setting up human feedback collection...")
        
        # Setup workflow if not done
        if not self.setup_completed:
            self.feedback_integration.setup_human_feedback_workflow()
            self.setup_completed = True
        
        # Prepare Dev B results for human review
        dev_b_evaluation_data = self._prepare_dev_b_for_human_review(
            candidate_prompt, dev_b_results
        )
        
        # Create traces for human feedback
        trace_ids = self.feedback_integration.process_dev_b_results_for_human_feedback(
            dev_b_results=dev_b_evaluation_data,
            candidate_prompt=candidate_prompt,
            baseline_metrics=baseline_metrics
        )
        
        # For now, return trace IDs for manual annotation
        # In a production system, you might wait for feedback or check periodically
        feedback_results = {
            "status": "traces_created",
            "trace_ids": trace_ids,
            "queue_name": self.feedback_integration.config.queue_name,
            "total_traces": len(trace_ids),
            "instructions": [
                "1. Go to LangFuse dashboard",
                "2. Navigate to 'Annotate' tab",
                "3. Add traces to annotation queue",
                "4. Complete human annotations"
            ]
        }
        
        print(f"✅ Created {len(trace_ids)} traces for human feedback")
        print("📝 Manual annotation required in LangFuse dashboard")
        
        return feedback_results
    
    def _prepare_dev_b_for_human_review(
        self,
        candidate_prompt: str,
        dev_b_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare Dev B evaluation results for human review
        
        Args:
            candidate_prompt: The optimized prompt
            dev_b_results: Dev B evaluation results
            
        Returns:
            List of evaluation data formatted for human review
        """
        evaluation_data = []
        
        # Extract failed cases for human review (most important)
        failed_cases = dev_b_results.get('detailed_failed_cases', {})
        wrong_classifications = failed_cases.get('wrong_classifications', [])
        
        # Limit to top 10 cases for human review
        for i, case in enumerate(wrong_classifications[:10]):
            evaluation_item = {
                "evaluation_id": f"dev_b_wrong_classification_{i}",
                "input_prompt": case.get('input_prompt', ''),
                "model_output": case.get('prediction_text', ''),
                "expected_output": case.get('ground_truth', {}),
                "failure_type": "wrong_classification",
                "wrong_fields": case.get('wrong_fields', []),
                "metadata": {
                    "example_idx": case.get('example_idx', i),
                    "timestamp": case.get('timestamp', ''),
                    "candidate_prompt_used": candidate_prompt
                }
            }
            evaluation_data.append(evaluation_item)
        
        print(f"📋 Prepared {len(evaluation_data)} cases for human review")
        return evaluation_data
    
    async def get_feedback_results(self, trace_ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve human feedback results for given trace IDs
        
        Args:
            trace_ids: List of trace IDs to get feedback for
            
        Returns:
            Human feedback summary
        """
        print(f"📊 Retrieving human feedback for {len(trace_ids)} traces...")
        
        try:
            feedback_results = self.feedback_manager.get_human_feedback_summary(trace_ids)
            return feedback_results
        except Exception as e:
            print(f"❌ Error retrieving human feedback: {e}")
            return {
                "status": "error",
                "error": str(e),
                "annotated_traces": 0,
                "total_traces": len(trace_ids)
            } 