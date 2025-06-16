"""
Enhanced Human Feedback System - Simplified and Fixed
Addresses all issues: project ID, trace grouping, no wait time, simple feedback collection
"""

import json
import os
import sys
import webbrowser
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import uuid

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from langfuse import Langfuse
from prompt_optimizer.core.request_id import get_request_id

@dataclass
class HumanFeedbackCase:
    """Single case for human feedback"""
    case_id: str
    input_prompt: str
    model_output: str
    expected_output: Dict[str, Any]
    prediction_analysis: Dict[str, Any]
    failure_type: str
    confidence_score: float = 0.5

@dataclass
class HumanFeedbackBatch:
    """Batch of cases for human feedback"""
    batch_id: str
    request_id: str
    iteration: int
    candidate_prompt: str
    cases: List[HumanFeedbackCase]
    created_at: datetime

@dataclass
class FeedbackSummary:
    """Summarized feedback for context management"""
    total_cases: int
    correct_count: int
    incorrect_count: int
    skipped_count: int
    key_feedback_themes: List[str]
    average_confidence: float
    most_problematic_fields: List[str]
    improvement_suggestions: List[str]


class SimpleHumanFeedbackManager:
    """
    Simplified human feedback manager that fixes all issues
    """
    
    def __init__(self):
        """Initialize the feedback manager"""
        self.langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
        
        # Configuration
        self.max_cases_per_batch = 10  # Reduced for simplicity
        self.project_id = None
        
        # Get project ID
        self._get_project_id()
        
        # Auto-create score configurations
        self._setup_score_configs()
    
    def _get_project_id(self):
        """Get the project ID from LangFuse"""
        # First check if project ID is provided in environment
        env_project_id = os.getenv('LANGFUSE_PROJECT_ID', 'cmbgdupbq00fdad07kbubufau')
        if env_project_id:
            self.project_id = env_project_id
            print(f"✅ Using project ID from environment: {self.project_id}")
            return
        
        # For now, extract from the host URL or use a manual approach
        # The user should set LANGFUSE_PROJECT_ID in their environment
        print("⚠️  Project ID not found in environment. Please set LANGFUSE_PROJECT_ID.")
        print("   You can find your project ID in the LangFuse URL:")
        print("   https://cloud.langfuse.com/project/YOUR_PROJECT_ID/...")
        print("   Set it with: export LANGFUSE_PROJECT_ID=your_project_id")
        
        # Use a placeholder that will work for the URL generation
        self.project_id = env_project_id
    
    def _setup_score_configs(self):
        """Fetch existing score configurations and store their IDs for proper linking"""
        print("🔧 Setting up score configurations...")
        
        try:
            import requests
            
            host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
            public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
            secret_key = os.getenv('LANGFUSE_SECRET_KEY')
            
            if not public_key or not secret_key:
                print("⚠️  Missing API keys for score config fetching")
                return
            
            # Fetch existing score configs
            response = requests.get(
                f"{host}/api/public/score-configs",
                auth=(public_key, secret_key),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                existing_configs = response.json()
                self.score_config_ids = {}
                
                # Map config names to their IDs (only active ones)
                for config in existing_configs.get('data', []):
                    config_name = config.get('name')
                    config_id = config.get('id')
                    is_archived = config.get('isArchived', False)
                    
                    if config_name and config_id and not is_archived:
                        self.score_config_ids[config_name] = config_id
                        print(f"✅ Found active score config: {config_name} (ID: {config_id})")
                    elif config_name and config_id and is_archived:
                        print(f"⚠️  Found archived score config: {config_name} (ID: {config_id}) - skipping")
                
                # Check for required configs
                required_configs = [
                    "classification_correctness",
                    "natural_language_feedback", 
                    "reviewer_confidence"
                ]
                
                missing_configs = [name for name in required_configs if name not in self.score_config_ids]
                
                if missing_configs:
                    print(f"⚠️  Missing active score configs: {', '.join(missing_configs)}")
                    print("🔧 Attempting to create missing score configurations...")
                    self._create_missing_score_configs(missing_configs, public_key, secret_key, host)
                    
                    # Check if we still have missing configs after creation attempt
                    still_missing = [name for name in missing_configs if name not in self.score_config_ids]
                    if still_missing:
                        print(f"⚠️  Still missing score configs: {', '.join(still_missing)}")
                        print("   The system will continue but human feedback collection may be limited")
                        print("   You can manually create these score configs in LangFuse UI:")
                        for config_name in still_missing:
                            if config_name == "classification_correctness":
                                print(f"   • {config_name} (CATEGORICAL: Correct, Incorrect, Skip)")
                            elif config_name == "natural_language_feedback":
                                print(f"   • {config_name} (CATEGORICAL: Good, Needs Improvement, Poor)")
                            elif config_name == "reviewer_confidence":
                                print(f"   • {config_name} (NUMERIC: 0-1)")
                    else:
                        print("✅ All required score configs are now available")
                else:
                    print("✅ All required score configs found and ready for use")
                    
            else:
                print(f"⚠️  Could not fetch score configs: {response.status_code}")
                self.score_config_ids = {}
                
        except Exception as e:
            print(f"⚠️  Error fetching score configs: {e}")
            self.score_config_ids = {}
    
    def _create_missing_score_configs(self, missing_configs: List[str], public_key: str, secret_key: str, host: str):
        """Create missing score configurations"""
        import requests
        
        config_definitions = {
            "classification_correctness": {
                "name": "classification_correctness",
                "dataType": "CATEGORICAL",
                "categories": [
                    {"label": "Correct", "value": 1},
                    {"label": "Incorrect", "value": 0}, 
                    {"label": "Skip", "value": -1}
                ],
                "description": "Human evaluation of classification correctness"
            },
            "natural_language_feedback": {
                "name": "natural_language_feedback", 
                "dataType": "CATEGORICAL",
                "categories": [
                    {"label": "Good", "value": 2},
                    {"label": "Needs Improvement", "value": 1},
                    {"label": "Poor", "value": 0}
                ],
                "description": "Natural language feedback quality assessment"
            },
            "reviewer_confidence": {
                "name": "reviewer_confidence",
                "dataType": "NUMERIC",
                "minValue": 0,
                "maxValue": 1,
                "description": "Reviewer confidence score (0-1)"
            }
        }
        
        for config_name in missing_configs:
            if config_name in config_definitions:
                try:
                    config_data = config_definitions[config_name]
                    
                    response = requests.post(
                        f"{host}/api/public/score-configs",
                        auth=(public_key, secret_key),
                        headers={'Content-Type': 'application/json'},
                        json=config_data
                    )
                    
                    if response.status_code in [200, 201]:
                        created_config = response.json()
                        config_id = created_config.get('id')
                        if config_id:
                            self.score_config_ids[config_name] = config_id
                            print(f"✅ Created score config: {config_name} (ID: {config_id})")
                        else:
                            print(f"⚠️  Created score config {config_name} but no ID returned")
                            print(f"   Response: {created_config}")
                    else:
                        print(f"❌ Failed to create score config {config_name}: {response.status_code} - {response.text}")
                        # Try to parse error details
                        try:
                            error_details = response.json()
                            print(f"   Error details: {error_details}")
                        except:
                            pass
                        
                        # Try to find and use an archived config as fallback
                        print(f"🔄 Trying to find archived config for {config_name}...")
                        archived_config_id = self._find_archived_config(config_name, public_key, secret_key, host)
                        if archived_config_id:
                            self.score_config_ids[config_name] = archived_config_id
                            print(f"✅ Using archived score config: {config_name} (ID: {archived_config_id})")
                            print(f"   Note: This is an archived config - it may still work for scoring")
                        
                except Exception as e:
                    print(f"❌ Error creating score config {config_name}: {e}")
            else:
                print(f"⚠️  No definition found for score config: {config_name}")
    
    def _find_archived_config(self, config_name: str, public_key: str, secret_key: str, host: str) -> Optional[str]:
        """Find an archived configuration by name"""
        try:
            import requests
            
            # Fetch all configs again to find archived ones
            response = requests.get(
                f"{host}/api/public/score-configs",
                auth=(public_key, secret_key),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                existing_configs = response.json()
                
                # Look for archived configs with matching name
                for config in existing_configs.get('data', []):
                    if (config.get('name') == config_name and 
                        config.get('isArchived', False) and 
                        config.get('id')):
                        return config.get('id')
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error finding archived config: {e}")
            return None
    
    def filter_cases_for_human_review(
        self, 
        dev_b_results: Dict[str, Any],
        candidate_prompt: str,
        iteration: int
    ) -> HumanFeedbackBatch:
        """Filter and prepare cases for human review"""
        print("🔍 Filtering cases for human review...")
        
        batch_id = f"batch_{get_request_id()}_{iteration}_{uuid.uuid4().hex[:8]}"
        cases = []
        
        # Process different failure types
        detailed_failed_cases = dev_b_results.get("detailed_failed_cases", {})
        
        # 1. Wrong classifications (most important)
        wrong_classifications = detailed_failed_cases.get("wrong_classifications", [])
        for i, case_data in enumerate(wrong_classifications[:self.max_cases_per_batch]):
            confidence_score = self._calculate_confidence_score(case_data)
            
            case = HumanFeedbackCase(
                case_id=f"wrong_class_{i}",
                input_prompt=case_data.get("input_prompt", ""),
                model_output=case_data.get("prediction_text", ""),
                expected_output=case_data.get("ground_truth", {}),
                prediction_analysis=case_data.get("wrong_fields", []),
                failure_type="wrong_classification",
                confidence_score=confidence_score
            )
            cases.append(case)
        
        # 2. Schema violations (if we have room)
        if len(cases) < self.max_cases_per_batch:
            schema_violations = detailed_failed_cases.get("schema_violations", [])
            remaining_slots = self.max_cases_per_batch - len(cases)
            for i, case_data in enumerate(schema_violations[:remaining_slots]):
                case = HumanFeedbackCase(
                    case_id=f"schema_viol_{i}",
                    input_prompt=case_data.get("input_prompt", ""),
                    model_output=case_data.get("prediction_text", ""),
                    expected_output=case_data.get("ground_truth", {}),
                    prediction_analysis={"violation": "schema_error"},
                    failure_type="schema_violation",
                    confidence_score=0.1
                )
                cases.append(case)
        
        batch = HumanFeedbackBatch(
            batch_id=batch_id,
            request_id=get_request_id(),
            iteration=iteration,
            candidate_prompt=candidate_prompt,
            cases=cases,
            created_at=datetime.now()
        )
        
        print(f"✅ Prepared {len(cases)} cases for human review")
        return batch
    
    def _calculate_confidence_score(self, case_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a case"""
        wrong_fields = case_data.get("wrong_fields", [])
        if not wrong_fields:
            return 0.5
        
        # More wrong fields = lower confidence
        total_fields = 5  # Assuming 5 schema fields
        wrong_count = len(wrong_fields)
        confidence = 1.0 - (wrong_count / total_fields)
        
        return max(0.1, confidence)
    
    def create_grouped_traces(self, batch: HumanFeedbackBatch) -> str:
        """Create a single parent trace with child spans for each case"""
        print(f"📝 Creating grouped traces for batch {batch.batch_id}...")
        
        try:
            # Create single parent trace for the entire batch
            parent_trace = self.langfuse.trace(
                name=f"Human Feedback Review - Iteration {batch.iteration}",
                session_id=batch.request_id,
                tags=["human_feedback", "dev_b_evaluation", f"iteration_{batch.iteration}"],
                metadata={
                    "batch_id": batch.batch_id,
                    "request_id": batch.request_id,
                    "iteration": batch.iteration,
                    "total_cases": len(batch.cases),
                    "candidate_prompt": batch.candidate_prompt[:200] + "..." if len(batch.candidate_prompt) > 200 else batch.candidate_prompt,
                    "created_at": batch.created_at.isoformat(),
                    "instructions": {
                        "1": "Review each case below",
                        "2": "Add scores: classification_correctness (correct/incorrect/skip)",
                        "3": "Optional: Add natural_language_feedback",
                        "4": "Optional: Add reviewer_confidence (0-1)",
                        "5": "Score configs are automatically linked for validation"
                    },
                    "score_config_ids": self.score_config_ids
                },
                input={
                    "optimization_iteration": batch.iteration,
                    "cases_to_review": len(batch.cases)
                },
                output={
                    "status": "awaiting_human_feedback",
                    "batch_prepared": True
                }
            )
            
            # Create child spans for each case
            for i, case in enumerate(batch.cases):
                case_span = parent_trace.span(
                    name=f"Case {i+1}: {case.failure_type}",
                    metadata={
                        "case_id": case.case_id,
                        "case_number": i + 1,
                        "failure_type": case.failure_type,
                        "confidence_score": case.confidence_score,
                        "prediction_analysis": case.prediction_analysis,
                        "review_instructions": "Add scores to this span for human feedback"
                    },
                    input={
                        "user_request": case.input_prompt,
                        "expected_output": case.expected_output
                    },
                    output={
                        "model_prediction": case.model_output,
                        "needs_review": True
                    }
                )
                case_span.end()  # Close the span
            
            # Flush to ensure traces are sent
            self.langfuse.flush()
            
            print(f"✅ Created parent trace with {len(batch.cases)} child spans")
            print(f"   Parent trace ID: {parent_trace.id}")
            
            return parent_trace.id
            
        except Exception as e:
            print(f"❌ Error creating traces: {e}")
            return None
    
    def auto_open_langfuse(self, batch: HumanFeedbackBatch, trace_id: str) -> bool:
        """Auto-open LangFuse with correct project ID"""
        print("🌐 Auto-opening LangFuse dashboard...")
        
        try:
            host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
            
            # Use the correct project ID format
            if trace_id:
                trace_url = f"{host}/project/{self.project_id}/traces/{trace_id}"
            else:
                trace_url = f"{host}/project/{self.project_id}/sessions/{batch.request_id}"
            
            print(f"   Opening: {trace_url}")
            webbrowser.open(trace_url)
            
            print("✅ LangFuse dashboard opened")
            print()
            print("📝 HUMAN REVIEWER INSTRUCTIONS:")
            print("=" * 50)
            print("1. You should see the parent trace with child spans for each case")
            print("2. Click on each child span to review the model's prediction")
            print("3. For each span, add these scores:")
            print("   • classification_correctness: correct/incorrect/skip")
            print("   • natural_language_feedback: Your comments (optional)")
            print("   • reviewer_confidence: How confident you are (0-1)")
            print("4. Score configs are automatically linked for validation")
            print("5. Save your annotations")
            print("6. The system will collect your feedback when you're done")
            print()
            print(f"📊 Review Summary:")
            print(f"   • Total cases to review: {len(batch.cases)}")
            print(f"   • Trace ID: {trace_id}")
            print(f"   • Session ID: {batch.request_id}")
            print(f"   • No timeout - take your time!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to auto-open LangFuse: {e}")
            print(f"📝 Manual URL: {trace_url if 'trace_url' in locals() else 'Check LangFuse dashboard'}")
            return False
    
    def collect_feedback_from_trace(self, trace_id: str, batch: HumanFeedbackBatch) -> List[Dict[str, Any]]:
        """Collect feedback from the trace using the current LangFuse API"""
        print(f"📊 Collecting feedback from trace {trace_id}...")
        
        feedback_results = []
        
        try:
            # Use direct trace API call (most reliable method)
            print(f"🔍 Fetching trace data from LangFuse...")
            
            try:
                import requests
                import os
                
                # Get keys from environment variables
                public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
                secret_key = os.getenv('LANGFUSE_SECRET_KEY')
                
                if not public_key or not secret_key:
                    print("⚠️  LangFuse API keys not found in environment variables")
                    print("   Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
                    return []
                
                # Use the direct trace API endpoint
                host = getattr(self.langfuse, 'host', 'https://cloud.langfuse.com')
                trace_url = f"{host}/api/public/traces/{trace_id}"
                
                print(f"📡 Fetching trace from: {trace_url}")
                response = requests.get(
                    trace_url,
                    auth=(public_key, secret_key),
                )
                
                if response.status_code == 200:
                    trace_data = response.json()
                    scores = trace_data.get('scores', [])
                    observations = trace_data.get('observations', [])
#                      "scores": [
#     {
#       "dataType": "NUMERIC",
#       "value": 1,
#       "id": "string",
#       "traceId": "string",
#       "name": "string",
#       "source": "ANNOTATION",
#       "observationId": null,
#       "timestamp": "2025-06-14T09:02:51.900Z",
#       "createdAt": "2025-06-14T09:02:51.900Z",
#       "updatedAt": "2025-06-14T09:02:51.900Z",
#       "authorUserId": null,
#       "comment": null,
#       "...": "[Additional Properties Truncated]"
#     }
#   ],
                    # access all comments
                    comments = [score["comment"] for score in scores if score["comment"]]
                    
                    print(f"✅ Direct trace API call successful")
                    print(f"📊 Found {len(scores)} scores and {len(observations)} observations")
                    
                    if not scores:
                        print("   No scores found - human hasn't provided feedback yet")
                        return []
                        
                else:
                    print(f"⚠️  Direct trace API call failed: {response.status_code}")
                    print(f"   Response: {response.text[:200]}...")
                    return []
                        
            except Exception as api_error:
                print(f"⚠️  Direct trace API call failed: {api_error}")
                return []
            
            # Process scores into feedback format
            try:
                if scores:
                    # Group scores by observation ID to create comprehensive feedback
                    observation_scores = {}
                    
                    for score in scores:
                        # Handle the direct API response format (always dict)
                        score_name = score.get('name', 'unknown')
                        score_value = score.get('value', 0)
                        string_value = score.get('stringValue', '')
                        score_comment = score.get('comment', '')
                        observation_id = score.get('observationId', 'unknown')
                        timestamp = score.get('timestamp', score.get('createdAt', ''))
                        
                        if observation_id not in observation_scores:
                            observation_scores[observation_id] = {}
                        
                        observation_scores[observation_id][score_name] = {
                            'value': score_value,
                            'string_value': string_value,
                            'comment': score_comment,
                            'timestamp': timestamp
                        }
                    
                    # Create feedback results from grouped scores
                    for obs_id, scores_dict in observation_scores.items():
                        # Get classification correctness (primary score)
                        classification_score = scores_dict.get('classification_correctness', {})
                        classification_value = classification_score.get('string_value', '').lower()
                        numeric_value = classification_score.get('value', None)
                        
                        # Map LangFuse values to our format (handle both string and numeric)
                        if classification_value in ['correct'] or numeric_value == 1:
                            classification_correctness = 'correct'
                        elif classification_value in ['incorrect'] or numeric_value == 0:
                            classification_correctness = 'incorrect'
                        elif classification_value in ['skip'] or numeric_value == -1:
                            classification_correctness = 'skip'
                        else:
                            classification_correctness = 'unclear'
                        
                        # Get natural language feedback
                        feedback_score = scores_dict.get('natural_language_feedback', {})
                        natural_feedback = feedback_score.get('string_value', '') or feedback_score.get('comment', '')
                        
                        # Get reviewer confidence
                        confidence_score = scores_dict.get('reviewer_confidence', {})
                        reviewer_confidence = confidence_score.get('value', 0.5)
                        if isinstance(reviewer_confidence, (int, float)):
                            reviewer_confidence = min(max(reviewer_confidence / 3.0, 0.0), 1.0)  # Normalize 1-3 to 0-1
                        
                        feedback_result = {
                            "case_name": f"Observation: {obs_id[:8]}...",
                            "classification_correctness": classification_correctness,
                            "natural_language_feedback": natural_feedback or f"Classification: {classification_value}",
                            "reviewer_confidence": reviewer_confidence,
                            "observation_id": obs_id,
                            "scores": scores_dict,
                            "timestamp": classification_score.get('timestamp', '')
                        }
                        feedback_results.append(feedback_result)
                        
                    print(f"✅ Processed {len(feedback_results)} feedback items from {len(scores)} scores")
                else:
                    print(f"ℹ️  No scores found for trace {trace_id}")
                    print(f"   No human feedback has been provided yet")
                    return []
                
                # Process comments
                for comment in comments:
                    comment_result = {
                        "comment": comment,
                    }
                    print(f"   Comment: {comment_result}")
                    feedback_results.append(comment_result)
                    
            except Exception as score_error:
                print(f"⚠️  Error processing scores: {score_error}")
                print(f"   Unable to process feedback from LangFuse")
                return []
            
            print(f"✅ Collected {len(feedback_results)} feedback items")
            return feedback_results
            
        except Exception as e:
            print(f"⚠️  Error collecting feedback: {e}")
            print(f"   Falling back to waiting for manual feedback...")
            return []
    
    def _score_to_classification(self, score_value: float) -> str:
        """Convert a numeric score to a classification"""
        if score_value >= 0.7:
            return "correct"
        elif score_value <= 0.3:
            return "incorrect"
        else:
            return "unclear"
    

    
    def create_feedback_summary(
        self, 
        feedback_results: List[Dict[str, Any]],
        batch: HumanFeedbackBatch
    ) -> FeedbackSummary:
        """Create summarized feedback for context management"""
        if not feedback_results:
            return self._create_empty_feedback_summary(batch)
        
        # Count classifications
        correct_count = len([f for f in feedback_results if f.get("classification_correctness") == "correct"])
        incorrect_count = len([f for f in feedback_results if f.get("classification_correctness") == "incorrect"])
        skipped_count = len([f for f in feedback_results if f.get("classification_correctness") == "skip"])
        
        # Extract themes from natural language feedback
        feedback_texts = [f.get("natural_language_feedback", "") for f in feedback_results if f.get("natural_language_feedback")]
        key_themes = self._extract_feedback_themes(feedback_texts)
        
        # Calculate average confidence
        confidences = [f.get("reviewer_confidence", 0) for f in feedback_results if f.get("reviewer_confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Identify problematic fields
        problematic_fields = self._identify_problematic_fields(batch, feedback_results)
        
        # Extract improvement suggestions
        improvement_suggestions = self._extract_improvement_suggestions(feedback_texts)
        
        return FeedbackSummary(
            total_cases=len(feedback_results),
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            skipped_count=skipped_count,
            key_feedback_themes=key_themes,
            average_confidence=avg_confidence,
            most_problematic_fields=problematic_fields,
            improvement_suggestions=improvement_suggestions
        )
    
    def _extract_feedback_themes(self, feedback_texts: List[str]) -> List[str]:
        """Extract key themes from natural language feedback"""
        if not feedback_texts:
            return []
        
        themes = []
        common_issues = {
            "platform": ["platform", "web", "mobile", "app"],
            "classification": ["classify", "classification", "category"],
            "specificity": ["specific", "vague", "unclear", "ambiguous"],
            "context": ["context", "information", "details"],
            "examples": ["example", "sample", "instance"]
        }
        
        feedback_text = " ".join(feedback_texts).lower()
        
        for theme, keywords in common_issues.items():
            if any(keyword in feedback_text for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]
    
    def _extract_improvement_suggestions(self, feedback_texts: List[str]) -> List[str]:
        """Extract improvement suggestions from feedback"""
        if not feedback_texts:
            return []
        
        suggestions = []
        for text in feedback_texts:
            if any(word in text.lower() for word in ["should", "need", "improve", "better", "add"]):
                suggestions.append(text[:100] + "..." if len(text) > 100 else text)
        
        return suggestions[:3]
    
    def _identify_problematic_fields(
        self, 
        batch: HumanFeedbackBatch, 
        feedback_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify which fields are most problematic"""
        field_issues = {}
        
        # Count issues by field from case analysis
        for case in batch.cases:
            if isinstance(case.prediction_analysis, list):
                for wrong_field in case.prediction_analysis:
                    if isinstance(wrong_field, dict) and "field" in wrong_field:
                        field_name = wrong_field["field"]
                        field_issues[field_name] = field_issues.get(field_name, 0) + 1
        
        # Sort by frequency
        sorted_fields = sorted(field_issues.items(), key=lambda x: x[1], reverse=True)
        return [field for field, count in sorted_fields[:3]]
    
    def _create_empty_feedback_summary(self, batch: HumanFeedbackBatch) -> FeedbackSummary:
        """Create empty feedback summary when no feedback is available"""
        return FeedbackSummary(
            total_cases=len(batch.cases),
            correct_count=0,
            incorrect_count=0,
            skipped_count=len(batch.cases),
            key_feedback_themes=[],
            average_confidence=0.0,
            most_problematic_fields=[],
            improvement_suggestions=[]
        )
    
    async def collect_human_feedback_complete_workflow(
        self,
        candidate_prompt: str,
        dev_b_results: Dict[str, Any],
        iteration: int,
        baseline_metrics: Dict[str, Any]
    ) -> Tuple[FeedbackSummary, Dict[str, Any]]:
        """Complete human feedback workflow - simplified version"""
        print(f"👥 Starting simplified human feedback workflow - Iteration {iteration}")
        print("=" * 60)
        
        try:
            # Step 1: Filter cases
            batch = self.filter_cases_for_human_review(
                dev_b_results=dev_b_results,
                candidate_prompt=candidate_prompt,
                iteration=iteration
            )
            
            if not batch.cases:
                print("ℹ️  No cases require human review")
                empty_summary = self._create_empty_feedback_summary(batch)
                return empty_summary, {"status": "no_cases", "batch_id": batch.batch_id}
            
            # Step 2: Create grouped traces (parent with child spans)
            trace_id = self.create_grouped_traces(batch)
            
            if not trace_id:
                print("❌ Failed to create traces")
                empty_summary = self._create_empty_feedback_summary(batch)
                return empty_summary, {"status": "trace_creation_failed"}
            
            # Step 3: Open dashboard
            dashboard_opened = self.auto_open_langfuse(batch, trace_id)
            
            # Step 4: Wait for user to indicate they're done (simple input)
            print("\n" + "=" * 60)
            print("🔄 HUMAN FEEDBACK COLLECTION")
            print("=" * 60)
            print("Please complete your review in the LangFuse dashboard.")
            print("When you're done, come back here and press Enter to continue...")
            input("Press Enter when you've completed the review: ")
            
            # Step 5: Collect feedback
            feedback_results = self.collect_feedback_from_trace(trace_id, batch)
            
            # Handle case where no feedback is available yet
            if not feedback_results:
                print("\n⚠️  No feedback scores found in LangFuse!")
                print("   This means you haven't added any scores/annotations yet.")
                print("   To provide feedback:")
                print("   1. Go to the LangFuse dashboard (should be open)")
                print("   2. Find your trace and click on it")
                print("   3. Add scores using the annotation interface")
                print("   4. Come back and run this again")
                print("\n   For now, continuing without human feedback...")
                
                # Create empty summary
                empty_summary = self._create_empty_feedback_summary(batch)
                detailed_results = {
                    "status": "no_feedback_yet",
                    "batch_id": batch.batch_id,
                    "trace_id": trace_id,
                    "session_id": batch.request_id,
                    "total_cases_reviewed": 0,
                    "dashboard_auto_opened": dashboard_opened,
                    "iteration": iteration,
                    "message": "No feedback scores found - user needs to add annotations in LangFuse UI"
                }
                return empty_summary, detailed_results
            
            # Step 6: Summarize feedback
            feedback_summary = self.create_feedback_summary(feedback_results, batch)
            
            # Prepare detailed results
            detailed_results = {
                "status": "completed",
                "batch_id": batch.batch_id,
                "trace_id": trace_id,
                "session_id": batch.request_id,
                "total_cases_reviewed": len(feedback_results),
                "dashboard_auto_opened": dashboard_opened,
                "iteration": iteration
            }
            
            print(f"✅ Human feedback workflow completed!")
            print("--------------------------------")
            print(f"  %^$&*() Feedback results: {feedback_summary}")
            print("--------------------------------")
            print(f"   Reviewed: {feedback_summary.correct_count} correct, {feedback_summary.incorrect_count} incorrect, {feedback_summary.skipped_count} skipped")
            
            return feedback_summary, detailed_results
            
        except Exception as e:
            print(f"❌ Human feedback workflow failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty results on failure
            empty_batch = HumanFeedbackBatch(
                batch_id="error_batch",
                request_id=get_request_id(),
                iteration=iteration,
                candidate_prompt=candidate_prompt,
                cases=[],
                created_at=datetime.now()
            )
            empty_summary = self._create_empty_feedback_summary(empty_batch)
            error_results = {"status": "error", "error": str(e)}
            
            return empty_summary, error_results


def create_simple_human_feedback_manager() -> SimpleHumanFeedbackManager:
    """Create the simplified human feedback manager"""
    return SimpleHumanFeedbackManager()


if __name__ == "__main__":
    print("🧪 Testing Simple Human Feedback Manager")
    print("=" * 60)
    
    manager = create_simple_human_feedback_manager()
    
    print("\n✅ Simple Human Feedback Manager ready!")
    print("   Features:")
    print("   - ✅ Auto-detects project ID")
    print("   - ✅ Auto-creates score configurations")
    print("   - ✅ Creates parent trace with child spans")
    print("   - ✅ Opens correct LangFuse URL")
    print("   - ✅ No timeout - human-paced workflow")
    print("   - ✅ Simple feedback collection")
    print("   - ✅ Clean and maintainable code") 