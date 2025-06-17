import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from langfuse import Langfuse
import httpx
import traceback
from datetime import datetime
import pymongo
from pymongo import MongoClient
import pytz
import dotenv

dotenv.load_dotenv()

try:
    import sys
    import os
    # Add fastapi_optimization_system to path if not already there
    fastapi_path = os.path.join(os.path.dirname(__file__), '..', '..', 'fastapi_optimization_system')
    if os.path.exists(fastapi_path) and fastapi_path not in sys.path:
        sys.path.insert(0, fastapi_path)
    
    from app.services.groq_service import get_groq_service
    GROQ_SERVICE_AVAILABLE = True
    print("✅ Groq service loaded successfully in metric.py")
except ImportError as e:
    print(f"Warning: Groq service not available in metric.py, falling back to direct API calls: {e}")
    GROQ_SERVICE_AVAILABLE = False

class JSONGenerationEvaluator:
    """
    Enhanced evaluator for JSON generation tasks with multi-layered validation.
    Designed specifically for prompt optimization workflows.
    
    Evaluates:
    1. JSON structure validity
    2. Schema compliance  
    3. Exact match accuracy
    4. Per-field enum classification metrics (Precision/Recall/F1)
    5. Failed case tracking for optimization
    6. Field-specific error patterns
    7. Optimization opportunity identification
    """
    
    def __init__(self, schema: Dict[str, List[str]]):
        """
        Initialize evaluator with simplified schema.
        
        Args:
            schema: Dictionary mapping field names to lists of possible enum values
        """
        self.schema = schema
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["personal_project_log_usage"]
        self.collection = self.db["llm_usage"]
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all internal metrics counters."""
        # Binary validation tallies
        self.tallies = {
            "valid_json_structure": {"correct": 0, "total": 0},
            "exact_json_match": {"correct": 0, "total": 0},
            "schema_compliance": {"correct": 0, "total": 0},
            "all_fields_present": {"correct": 0, "total": 0}
        }
        
        # Confusion matrix storage for enum fields
        self.enum_confusion = {}
        
        # Failed cases tracking for optimization
        self.failed_cases = {
            "invalid_json": [],           # Cases where JSON parsing failed
            "schema_violations": [],      # Cases with invalid enum values
            "missing_fields": [],         # Cases with missing required fields
            "wrong_classifications": [],  # Cases with wrong but valid enum values
            # "exact_match_failures": []    # Cases that failed exact match
        }
        
        # Field-specific error patterns
        self.field_error_patterns = {}
        
        # Initialize confusion matrices for enum fields
        for field, enum_values in self.schema.items():
            self.enum_confusion[field] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
            self.field_error_patterns[field] = {
                "missing_count": 0,
                "invalid_values": defaultdict(int),
                "confusion_pairs": defaultdict(int)  # actual -> predicted pairs
            }
    
    def evaluate_batch(self, 
                      ground_truth_jsons: List[Dict[str, Any]], 
                      predicted_texts: List[str],
                      input_prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions against references.
        
        Args:
            ground_truth_jsons: List of ground truth JSON objects
            predicted_texts: List of predicted JSON strings
            input_prompts: Optional list of input prompts for failed case analysis
            
        Returns:
            Dictionary containing all computed metrics and failed cases
        """
        if len(ground_truth_jsons) != len(predicted_texts):
            raise ValueError("Reference and prediction lists must have same length")
        
        if input_prompts and len(input_prompts) != len(predicted_texts):
            raise ValueError("Input prompts and prediction lists must have same length")
        
        self.reset_metrics()
        
        for i, (ref_json, pred_text) in enumerate(zip(ground_truth_jsons, predicted_texts)):
            input_prompt = input_prompts[i] if input_prompts else f"Example {i+1}"
            self._evaluate_single_example(ref_json, pred_text, input_prompt, i)
        
        return self._compute_final_metrics()
    
    def _evaluate_single_example(self, ref_json: Dict[str, Any], pred_text: str, input_prompt: str, example_idx: int):
        """Evaluate a single prediction example with detailed failure tracking."""
        
        example_data = {
            "example_idx": example_idx,
            "input_prompt": input_prompt,
            "ground_truth": ref_json,
            "prediction_text": pred_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # Clean up markdown code block formatting if present
        cleaned_pred_text = self._clean_prediction_text(pred_text)
        
        # 1. Check JSON structure validity
        try:
            parsed_pred = json.loads(cleaned_pred_text)
            self.tallies["valid_json_structure"]["correct"] += 1
        except json.JSONDecodeError as e:
            parsed_pred = None
            self.failed_cases["invalid_json"].append({
                **example_data,
                "error": str(e),
                "cleaned_text": cleaned_pred_text
            })
        self.tallies["valid_json_structure"]["total"] += 1
        
        # 2. Check exact JSON match
        if parsed_pred == ref_json:
            self.tallies["exact_json_match"]["correct"] += 1
        # else:
        #     self.failed_cases["exact_match_failures"].append({
        #         **example_data,
        #         "parsed_prediction": parsed_pred
        #     })
        self.tallies["exact_json_match"]["total"] += 1
        
        # 3. Detailed evaluation only if JSON is valid and is a dictionary
        if parsed_pred is not None and isinstance(parsed_pred, dict):
            self._evaluate_fields_detailed(ref_json, parsed_pred, example_data)
            
            # Check schema compliance
            if self._is_schema_compliant(parsed_pred):
                self.tallies["schema_compliance"]["correct"] += 1
            self.tallies["schema_compliance"]["total"] += 1
            
            # Check if all required fields are present
            if self._all_fields_present(parsed_pred):
                self.tallies["all_fields_present"]["correct"] += 1
            self.tallies["all_fields_present"]["total"] += 1
        elif parsed_pred is not None and not isinstance(parsed_pred, dict):
            # Handle case where JSON parsing succeeded but didn't return a dict
            self.failed_cases["invalid_json"].append({
                **example_data,
                "error": f"Parsed JSON is not a dictionary, got {type(parsed_pred).__name__}: {parsed_pred}",
                "cleaned_text": cleaned_pred_text
            })
    
    def _clean_prediction_text(self, pred_text: str) -> str:
        """Clean and extract JSON from prediction text."""
        if pred_text.startswith("```json") or pred_text.startswith("```"):
            try:
                start_idx = pred_text.find("\n", pred_text.find("```")) + 1
                end_idx = pred_text.rfind("```")
                if start_idx > 0 and end_idx > start_idx:
                    return pred_text[start_idx:end_idx].strip()
            except:
                pass
        return pred_text.strip()
    
    def _is_schema_compliant(self, pred_json: Dict[str, Any]) -> bool:
        """Check if prediction follows schema constraints."""
        if not isinstance(pred_json, dict):
            return False
        for field, enum_values in self.schema.items():
            if field in pred_json:
                pred_val = pred_json[field]
                if isinstance(pred_val, list):
                    for item in pred_val:
                        try:
                            if item not in enum_values:
                                return False
                        except TypeError:
                            # Unhashable type, consider it invalid
                            return False
                else:
                    try:
                        if pred_val not in enum_values:
                            return False
                    except TypeError:
                        # Unhashable type, consider it invalid
                        return False
        return True
    
    def _all_fields_present(self, pred_json: Dict[str, Any]) -> bool:
        """Check if all required fields are present."""
        if not isinstance(pred_json, dict):
            return False
        return all(field in pred_json for field in self.schema.keys())
    
    def _evaluate_fields_detailed(self, ref_json: Dict[str, Any], pred_json: Dict[str, Any], example_data: Dict[str, Any]):
        """Evaluate individual fields with detailed error tracking."""
        
        if not isinstance(pred_json, dict):
            print(f"⚠️ Warning: pred_json is not a dictionary, got {type(pred_json).__name__}: {pred_json}")
            return
        
        wrong_fields_for_example = []
        
        for field, enum_values in self.schema.items():
            pred_val = pred_json.get(field)
            ref_val = ref_json.get(field)
            
            if field not in pred_json:
                self.field_error_patterns[field]["missing_count"] += 1
                self.failed_cases["missing_fields"].append({
                    **example_data,
                    "missing_field": field,
                    "expected_value": ref_val
                })
                continue

            pred_vals = pred_val if isinstance(pred_val, list) else [pred_val]
            invalid_vals = []
            for v in pred_vals:
                if v is None:
                    continue
                try:
                    if v not in enum_values:
                        invalid_vals.append(v)
                except TypeError:
                    # Handle unhashable types (like dict, list)
                    invalid_vals.append(v)

            if invalid_vals:
                key_val = str(pred_val)
                self.field_error_patterns[field]["invalid_values"][key_val] += 1
                self.failed_cases["schema_violations"].append({
                    **example_data,
                    "field": field,
                    "invalid_value": pred_val,
                    "error_details": f"The following values are not in schema: {invalid_vals}",
                    "expected_value": ref_val,
                    "valid_options": enum_values
                })
                continue

            ref_vals = ref_val if isinstance(ref_val, list) else [ref_val]
            
            # Convert to hashable types for comparison
            def make_hashable(val):
                if isinstance(val, (dict, list)):
                    return str(val)
                return val
            
            pred_vals_hashable = [make_hashable(v) for v in pred_vals]
            ref_vals_hashable = [make_hashable(v) for v in ref_vals]
            
            if set(pred_vals_hashable) != set(ref_vals_hashable):
                all_refs_valid = True
                for v in ref_vals:
                    if v is None:
                        continue
                    try:
                        if v not in enum_values:
                            all_refs_valid = False
                            break
                    except TypeError:
                        all_refs_valid = False
                        break
                if all_refs_valid:
                    self.field_error_patterns[field]["confusion_pairs"][f"{sorted([str(v) for v in ref_vals])} -> {sorted([str(v) for v in pred_vals])}"] += 1
                    wrong_fields_for_example.append({
                        "field": field,
                        "predicted_value": pred_vals,
                        "expected_value": ref_vals
                    })

            self._update_enum_confusion(field, ref_vals, pred_vals, enum_values)
        
        if wrong_fields_for_example:
            grouped_wrong_classification = {
                **example_data,
                "wrong_fields": wrong_fields_for_example
            }
            self.failed_cases["wrong_classifications"].append(grouped_wrong_classification)
    
    def _update_enum_confusion(self, field: str, ref_vals: list, pred_vals: list, enum_values: List[str]):
        """Update confusion matrix for a single field based on reference and prediction lists."""
        # Filter out unhashable types and None values for set operations
        def filter_hashable(vals):
            hashable_vals = []
            for v in vals:
                if v is None:
                    continue
                try:
                    # Test if value is hashable by trying to add to set
                    {v}
                    hashable_vals.append(v)
                except TypeError:
                    # Skip unhashable types
                    continue
            return hashable_vals
        
        ref_set = set(filter_hashable(ref_vals))
        pred_set = set(filter_hashable(pred_vals))
        
        for label in enum_values:
            is_in_ref = label in ref_set
            is_in_pred = label in pred_set
            
            if is_in_ref and is_in_pred:
                self.enum_confusion[field][label]["TP"] += 1
            elif is_in_pred and not is_in_ref:
                self.enum_confusion[field][label]["FP"] += 1
            elif is_in_ref and not is_in_pred:
                self.enum_confusion[field][label]["FN"] += 1
    
    def _compute_final_metrics(self) -> Dict[str, Any]:
        """Compute all final metrics from collected statistics."""
        
        results = {}
        
        # Enhanced validation metrics
        results["validation_metrics"] = {
            "valid_json_accuracy": self._safe_divide(
                self.tallies["valid_json_structure"]["correct"],
                self.tallies["valid_json_structure"]["total"]
            ),
            "exact_match_accuracy": self._safe_divide(
                self.tallies["exact_json_match"]["correct"],
                self.tallies["exact_json_match"]["total"]
            ),
            "schema_compliance_accuracy": self._safe_divide(
                self.tallies["schema_compliance"]["correct"],
                self.tallies["schema_compliance"]["total"]
            ),
            "all_fields_present_accuracy": self._safe_divide(
                self.tallies["all_fields_present"]["correct"],
                self.tallies["all_fields_present"]["total"]
            )
        }
        
        # Enum field metrics
        results["enum_field_metrics"] = {}
        for field, confusion_matrix in self.enum_confusion.items():
            results["enum_field_metrics"][field] = self._compute_enum_metrics(field, confusion_matrix)
        
        # Failed cases summary
        total_wrong_classifications = 0
        for case in self.failed_cases["wrong_classifications"]:
            if "wrong_fields" in case:
                total_wrong_classifications += len(case["wrong_fields"])
            else:
                total_wrong_classifications += 1  # Fallback for old format
        
        results["failed_cases_summary"] = {
            "invalid_json_count": len(self.failed_cases["invalid_json"]),
            "schema_violations_count": len(self.failed_cases["schema_violations"]),
            "missing_fields_count": len(self.failed_cases["missing_fields"]),
            "wrong_classifications_count": total_wrong_classifications,
            # "exact_match_failures_count": len(self.failed_cases["exact_match_failures"])
        }
        
        # Field-specific insights
        results["field_insights"] = self._compute_field_insights()
        
        # Optimization opportunities
        results["optimization_opportunities"] = self._identify_optimization_opportunities()
        
        # Overall summary
        results["summary"] = self._compute_summary_metrics(results)
        
        # Store detailed failed cases
        results["detailed_failed_cases"] = self.failed_cases
        
        return results
    
    def _compute_field_insights(self) -> Dict[str, Dict[str, Any]]:
        """Compute field-specific insights for optimization."""
        insights = {}
        
        for field, patterns in self.field_error_patterns.items():
            total_examples = self.tallies["valid_json_structure"]["total"]
            
            insights[field] = {
                "missing_rate": self._safe_divide(patterns["missing_count"], total_examples),
                "most_common_invalid_values": dict(sorted(
                    patterns["invalid_values"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]),  # Top 5 most common invalid values
                "most_common_confusions": dict(sorted(
                    patterns["confusion_pairs"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]),  # Top 5 most common confusion pairs
                "error_severity": self._calculate_field_error_severity(field)
            }
        
        return insights
    
    def _calculate_field_error_severity(self, field: str) -> str:
        """Calculate error severity for a field."""
        field_metrics = self.enum_confusion.get(field, {})
        if not field_metrics:
            return "HIGH"  # No correct predictions
        
        total_correct = sum(counts["TP"] for counts in field_metrics.values())
        total_examples = sum(counts["TP"] + counts["FN"] for counts in field_metrics.values())
        
        if total_examples == 0:
            return "UNKNOWN"
        
        accuracy = self._safe_divide(total_correct, total_examples)
        
        if accuracy >= 0.9:
            return "LOW"
        elif accuracy >= 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _identify_optimization_opportunities(self) -> Dict[str, List[str]]:
        """Identify specific optimization opportunities based on failure patterns."""
        opportunities = {
            "prompt_engineering": [],
            "schema_clarification": [],
            "example_enhancement": [],
            "instruction_refinement": []
        }
        
        # Check for JSON structure issues
        if self.tallies["valid_json_structure"]["correct"] < self.tallies["valid_json_structure"]["total"]:
            opportunities["prompt_engineering"].append(
                "Add explicit JSON formatting instructions and examples"
            )
        
        # Check for schema compliance issues
        if len(self.failed_cases["schema_violations"]) > 0:
            opportunities["schema_clarification"].append(
                "Provide clearer enum value definitions and constraints"
            )
        
        # Check for missing fields
        if len(self.failed_cases["missing_fields"]) > 0:
            opportunities["instruction_refinement"].append(
                "Emphasize requirement for all fields to be present"
            )
        
        # Check field-specific patterns
        for field, patterns in self.field_error_patterns.items():
            if patterns["missing_count"] > 0:
                opportunities["instruction_refinement"].append(
                    f"Add specific guidance for '{field}' field"
                )
            
            if patterns["invalid_values"]:
                opportunities["example_enhancement"].append(
                    f"Provide more examples for '{field}' field classifications"
                )
        
        return opportunities
    
    def _compute_enum_metrics(self, field: str, confusion_matrix: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Compute precision, recall, F1 for an enum field with per-class breakdowns."""
        
        if not confusion_matrix:
            return {
                "macro_precision": 0.0, 
                "macro_recall": 0.0, 
                "macro_f1": 0.0,
                "micro_precision": 0.0, 
                "micro_recall": 0.0, 
                "micro_f1": 0.0, 
                "accuracy": 0.0,
                "per_class": {}
            }
        
        # Per-class metrics
        per_class_metrics = {}
        precisions, recalls, f1s = [], [], []
        total_tp = total_fp = total_fn = 0
        
        for class_name, counts in confusion_matrix.items():
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            total_tp += tp
            total_fp += fp  
            total_fn += fn
            
            # Per-class precision, recall, F1
            prec = self._safe_divide(tp, tp + fp)
            rec = self._safe_divide(tp, tp + fn)
            f1 = self._safe_divide(2 * prec * rec, prec + rec)
            
            # Store class-specific metrics
            per_class_metrics[class_name] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": tp + fn,  # Total examples of this class
                "correct": tp,       # Correctly predicted examples
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        
        # Macro averages (average across classes)
        macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        
        # Micro averages (aggregate then compute)
        micro_precision = self._safe_divide(total_tp, total_tp + total_fp)
        micro_recall = self._safe_divide(total_tp, total_tp + total_fn)
        micro_f1 = self._safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        
        # Accuracy
        accuracy = self._safe_divide(total_tp, total_tp + total_fn)
        
        return {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "accuracy": accuracy,
            "per_class": per_class_metrics
        }
    
    def _compute_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute high-level summary metrics."""
        
        # Average enum field F1 scores
        enum_f1_scores = []
        enum_precision_scores = []
        enum_recall_scores = []
        
        for field_metrics in results["enum_field_metrics"].values():
            enum_f1_scores.append(field_metrics["macro_f1"])
            enum_precision_scores.append(field_metrics["macro_precision"])
            enum_recall_scores.append(field_metrics["macro_recall"])
        
        return {
            "average_enum_macro_f1": sum(enum_f1_scores) / len(enum_f1_scores) if enum_f1_scores else 0.0,
            "average_enum_macro_precision": sum(enum_precision_scores) / len(enum_precision_scores) if enum_precision_scores else 0.0,
            "average_enum_macro_recall": sum(enum_recall_scores) / len(enum_recall_scores) if enum_recall_scores else 0.0,
            "total_examples": self.tallies["valid_json_structure"]["total"]
        }
    
    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safe division that returns 0.0 for division by zero."""
        return numerator / denominator if denominator > 0 else 0.0
    
    def print_detailed_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report with percentages and per-class metrics."""
        
        print("=" * 80)
        print("📊 JSON GENERATION EVALUATION REPORT")
        print("=" * 80)
        
        # Validation metrics
        print("\n📋 VALIDATION METRICS:")
        vm = results["validation_metrics"]
        print(f"  Valid JSON Structure:     {vm['valid_json_accuracy']*100:.2f}%")
        print(f"  Exact Match:              {vm['exact_match_accuracy']*100:.2f}%")
        
        # Enum field metrics
        if results["enum_field_metrics"]:
            print("\n🎯 FIELD METRICS:")
            for field, metrics in results["enum_field_metrics"].items():
                print(f"\n  📌 {field.upper()}:")
                print(f"    Macro Precision:  {metrics['macro_precision']*100:.2f}%")
                print(f"    Macro Recall:     {metrics['macro_recall']*100:.2f}%")
                print(f"    Macro F1:         {metrics['macro_f1']*100:.2f}%")
                print(f"    Micro Precision:  {metrics['micro_precision']*100:.2f}%")
                print(f"    Micro Recall:     {metrics['micro_recall']*100:.2f}%")
                print(f"    Micro F1:         {metrics['micro_f1']*100:.2f}%")
                print(f"    Accuracy:         {metrics['accuracy']*100:.2f}%")
                
                # Per-class metrics
                print(f"\n    CLASS-LEVEL METRICS:")
                print(f"    {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10} {'Accuracy':<10}")
                print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
                
                for class_name, class_metrics in metrics["per_class"].items():
                    precision = class_metrics["precision"] * 100
                    recall = class_metrics["recall"] * 100
                    f1 = class_metrics["f1"] * 100
                    support = class_metrics["support"]
                    accuracy = self._safe_divide(class_metrics["tp"], support) * 100
                    
                    print(f"    {class_name:<15} {precision:<11.2f}% {recall:<11.2f}% {f1:<11.2f}% {support:<10} {accuracy:<9.2f}%")
        
        # Summary
        print("\n📈 OVERALL SUMMARY:")
        summary = results["summary"]
        print(f"  Average Macro Precision:  {summary['average_enum_macro_precision']*100:.2f}%")
        print(f"  Average Macro Recall:     {summary['average_enum_macro_recall']*100:.2f}%")
        print(f"  Average Macro F1:         {summary['average_enum_macro_f1']*100:.2f}%")
        print(f"  Total Examples:           {summary['total_examples']}")
        
        # Confusion matrix in a readable format
        print("\n🧩 CONFUSION MATRICES:")
        for field, confusion in self.enum_confusion.items():
            if not confusion:
                continue
                
            print(f"\n  Field: {field.upper()}")
            class_names = sorted(list(confusion.keys()))
            
            # Calculate totals for each true class
            class_totals = {}
            for class_name, counts in confusion.items():
                class_totals[class_name] = counts["TP"] + counts["FN"]
            
            # Print the confusion matrix header
            print(f"  {'TRUE PRED':<15}", end="")
            for pred_class in class_names:
                print(f"{pred_class:<15}", end="")
            print("Total")
            
            # Print each row
            for true_class in class_names:
                print(f"  {true_class:<15}", end="")
                true_total = class_totals[true_class]
                
                for pred_class in class_names:
                    if true_class == pred_class:
                        # Correctly predicted as this class
                        value = confusion[true_class]["TP"]
                        percentage = self._safe_divide(value, true_total) * 100
                        print(f"{value} ({percentage:.1f}%)".ljust(15), end="")
                    else:
                        # We need to determine how many examples of true_class were predicted as pred_class
                        # This requires additional tracking, so we'll estimate based on available data
                        # In a full implementation, you'd track a full confusion matrix
                        value = 0  # Placeholder - ideally you'd track the full confusion matrix
                        percentage = 0.0
                        print(f"{value} ({percentage:.1f}%)".ljust(15), end="")
                
                # Print row total
                print(f"{true_total}")
        
        print("=" * 80)

    def save_failed_cases(self, filepath: str, format: str = "json"):
        """
        Save failed cases to file for optimization analysis.
        
        Args:
            filepath: Path to save the failed cases
            format: Format to save in ('json' or 'csv')
        """
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save to metrics directory
        filepath = os.path.join(metrics_dir, os.path.basename(filepath))
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(self.failed_cases, f, indent=2)
        elif format == "csv":
            # Flatten failed cases for CSV export
            all_failures = []
            for failure_type, cases in self.failed_cases.items():
                for case in cases:
                    if failure_type == "wrong_classifications" and "wrong_fields" in case:
                        # Handle grouped wrong classifications
                        base_case = {k: v for k, v in case.items() if k != "wrong_fields"}
                        base_case["failure_type"] = failure_type
                        
                        for wrong_field in case["wrong_fields"]:
                            field_case = base_case.copy()
                            field_case.update(wrong_field)
                            all_failures.append(field_case)
                    else:
                        # Handle other failure types normally
                        case_flat = case.copy()
                        case_flat["failure_type"] = failure_type
                        all_failures.append(case_flat)
            
            df = pd.DataFrame(all_failures)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError("Format must be 'json' or 'csv'")
        
        print(f"💾 Failed cases saved to: {filepath}")
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed optimization report with actionable insights."""
        
        report = []
        report.append("=" * 80)
        report.append("🚀 PROMPT OPTIMIZATION ANALYSIS REPORT")
        report.append("=" * 80)
        
        # High-level performance summary
        vm = results["validation_metrics"]
        summary = results["summary"]
        
        report.append(f"\n📊 PERFORMANCE OVERVIEW:")
        report.append(f"  Valid JSON Rate:          {vm['valid_json_accuracy']*100:.1f}%")
        report.append(f"  Exact Match Accuracy:     {vm['exact_match_accuracy']*100:.1f}%")
        report.append(f"  Schema Compliance:        {vm['schema_compliance_accuracy']*100:.1f}%")
        report.append(f"  All Fields Present:       {vm['all_fields_present_accuracy']*100:.1f}%")
        report.append(f"  Average Macro F1:         {summary['average_enum_macro_f1']*100:.1f}%")
        
        # Critical issues identification
        report.append(f"\n🚨 CRITICAL ISSUES:")
        fc_summary = results["failed_cases_summary"]
        
        if fc_summary["invalid_json_count"] > 0:
            report.append(f"  ❌ {fc_summary['invalid_json_count']} JSON parsing failures")
        if fc_summary["schema_violations_count"] > 0:
            report.append(f"  ⚠️  {fc_summary['schema_violations_count']} schema violations")
        if fc_summary["missing_fields_count"] > 0:
            report.append(f"  📝 {fc_summary['missing_fields_count']} missing field instances")
        if fc_summary["wrong_classifications_count"] > 0:
            report.append(f"  🎯 {fc_summary['wrong_classifications_count']} classification errors")
        
        # Field-specific insights
        report.append(f"\n🎯 FIELD-SPECIFIC INSIGHTS:")
        for field, insights in results["field_insights"].items():
            severity = insights["error_severity"]
            emoji = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
            
            report.append(f"\n  {emoji} {field.upper()} (Severity: {severity})")
            report.append(f"    Missing Rate: {insights['missing_rate']*100:.1f}%")
            
            if insights["most_common_invalid_values"]:
                report.append(f"    Common Invalid Values: {list(insights['most_common_invalid_values'].keys())[:3]}")
            
            if insights["most_common_confusions"]:
                top_confusion = list(insights["most_common_confusions"].keys())[0]
                report.append(f"    Top Confusion: {top_confusion}")
        
        # Optimization opportunities
        report.append(f"\n💡 OPTIMIZATION OPPORTUNITIES:")
        opportunities = results["optimization_opportunities"]
        
        for category, suggestions in opportunities.items():
            if suggestions:
                report.append(f"\n  📌 {category.replace('_', ' ').title()}:")
                for suggestion in suggestions[:3]:  # Top 3 suggestions per category
                    report.append(f"    • {suggestion}")
        
        # Performance recommendations
        report.append(f"\n🎯 PERFORMANCE RECOMMENDATIONS:")
        
        if vm["valid_json_accuracy"] < 0.95:
            report.append(f"  • URGENT: Improve JSON formatting (only {vm['valid_json_accuracy']*100:.1f}% valid)")
        
        if vm["schema_compliance_accuracy"] < 0.9:
            report.append(f"  • HIGH: Enhance schema compliance training")
        
        if summary["average_enum_macro_f1"] < 0.7:
            report.append(f"  • MEDIUM: Add more classification examples and clearer field definitions")
        
        # Priority fields for optimization
        high_priority_fields = [
            field for field, insights in results["field_insights"].items()
            if insights["error_severity"] == "HIGH"
        ]
        
        if high_priority_fields:
            report.append(f"\n🔥 PRIORITY FIELDS FOR OPTIMIZATION:")
            for field in high_priority_fields:
                field_metrics = results["enum_field_metrics"][field]
                report.append(f"  • {field}: {field_metrics['macro_f1']*100:.1f}% F1 Score")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def compare_metrics(baseline_results: Dict[str, Any], 
                       new_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metrics between baseline and new prompt results.
        
        Args:
            baseline_results: Results from baseline evaluation
            new_results: Results from new prompt evaluation
            
        Returns:
            Dictionary with comparison metrics and improvements
        """
        comparison = {
            "overall_improvement": {},
            "field_improvements": {},
            "regression_alerts": [],
            "significant_improvements": []
        }
        
        # Overall metrics comparison
        baseline_vm = baseline_results["validation_metrics"]
        new_vm = new_results["validation_metrics"]
        baseline_summary = baseline_results["summary"]
        new_summary = new_results["summary"]
        
        comparison["overall_improvement"] = {
            "valid_json_improvement": new_vm["valid_json_accuracy"] - baseline_vm["valid_json_accuracy"],
            "exact_match_improvement": new_vm["exact_match_accuracy"] - baseline_vm["exact_match_accuracy"],
            "schema_compliance_improvement": new_vm.get("schema_compliance_accuracy", 0) - baseline_vm.get("schema_compliance_accuracy", 0),
            "avg_f1_improvement": new_summary["average_enum_macro_f1"] - baseline_summary["average_enum_macro_f1"]
        }
        
        # Field-level comparison
        for field in baseline_results["enum_field_metrics"]:
            if field in new_results["enum_field_metrics"]:
                baseline_f1 = baseline_results["enum_field_metrics"][field]["macro_f1"]
                new_f1 = new_results["enum_field_metrics"][field]["macro_f1"]
                improvement = new_f1 - baseline_f1
                
                comparison["field_improvements"][field] = {
                    "f1_improvement": improvement,
                    "baseline_f1": baseline_f1,
                    "new_f1": new_f1
                }
                
                # Check for significant improvements (>5%)
                if improvement > 0.05:
                    comparison["significant_improvements"].append({
                        "field": field,
                        "improvement": improvement,
                        "improvement_percentage": (improvement / baseline_f1) * 100 if baseline_f1 > 0 else float('inf')
                    })
                
                # Check for regressions (>2% drop)
                elif improvement < -0.02:
                    comparison["regression_alerts"].append({
                        "field": field,
                        "regression": improvement,
                        "regression_percentage": (improvement / baseline_f1) * 100 if baseline_f1 > 0 else float('inf')
                    })
        
        return comparison
    
    def get_stratified_sample(self, failed_cases: Dict[str, List], 
                            sample_size: int = 10) -> Dict[str, List]:
        """
        Get a stratified sample of failed cases for human review.
        
        Args:
            failed_cases: Dictionary of failed cases by type
            sample_size: Total number of cases to sample
            
        Returns:
            Stratified sample of failed cases
        """
        import random
        
        total_failures = sum(len(cases) for cases in failed_cases.values())
        if total_failures == 0:
            return {}
        
        stratified_sample = {}
        
        for failure_type, cases in failed_cases.items():
            if not cases:
                continue
                
            # Calculate proportional sample size
            proportion = len(cases) / total_failures
            type_sample_size = max(1, int(sample_size * proportion))
            type_sample_size = min(type_sample_size, len(cases))
            
            # Random sample from this failure type
            stratified_sample[failure_type] = random.sample(cases, type_sample_size)
        
        return stratified_sample


def fetch_data_from_langfuse(dataset_name: str, limit: int = 2) -> List[Dict[str, Any]]:
    """
    Fetch ground truth data from LangFuse dataset.
    
    Args:
        dataset_name: Name of the dataset in LangFuse
        limit: Maximum number of examples to fetch
        
    Returns:
        List of ground truth JSON objects
    """

    langfuse_client = Langfuse(
    secret_key="sk-lf-d87cc28d-5a97-4fd9-bccd-13cfbf5e6ad3",
    public_key="pk-lf-4e626ffa-7bcd-495b-9f4d-f2f2c5b15087",
    host="https://cloud.langfuse.com"
    )
    
    # Fetch dataset examples
    try:
        print("Fetching dataset from LangFuse")
        dataset = langfuse_client.get_dataset(dataset_name)
        print(dataset)
        print("Fetching examples from LangFuse")
        
        # Extract ground truth data from examples
        ground_truth_data = []
        for i, item in enumerate(dataset.items):
            if i >= limit:  # Only process specified number of items
                break
                
            try:
                print(f"Processing example {item.input}")
                # Parse the JSON from the example
                if isinstance(item.expected_output, dict):
                    ground_truth_data.append(item.expected_output)
                elif isinstance(item.expected_output, str):
                    ground_truth_data.append(json.loads(item.expected_output))
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f"Error parsing example {item.id}: {e}")
                continue
        
        return ground_truth_data
    except Exception as e:
        print(f"Error fetching data from LangFuse: {e}")
        return []


def run_inference(prompts: List[str], base_prompt: str, model_config = None) -> List[str]:
    """
    Run inference using the configured model service or fallback to groq.
    
    Args:
        prompts: List of input prompts
        base_prompt: Base instruction prompt to use as system prompt
        model_config: Model configuration object with provider and model_name. If None, defaults to groq
        
    Returns:
        List of model responses
    """
    # For backward compatibility, default to groq if no model_config provided
    if model_config is None:
        return run_groq_inference(prompts, base_prompt, "llama-3.3-70b-versatile")
    
    # Try to use the model service factory if available
    try:
        import asyncio
        from app.services.model_service_factory import get_model_service_factory
        
        async def _async_inference():
            factory = get_model_service_factory()
            return await factory.batch_inference(
                model_config=model_config,
                prompts=prompts,
                system_prompt=base_prompt,
                temperature=getattr(model_config, 'temperature', 1.0),
                max_tokens=1024
            )
        
        # Try to run in existing event loop or create new one
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            print("Warning: run_inference called from async context, using fallback")
            # Fall through to provider-specific fallback
        except RuntimeError:
            # No running event loop, create one
            return asyncio.run(_async_inference())
    
    except ImportError:
        print("Model service factory not available, using provider-specific fallback")
    except Exception as e:
        print(f"Error using model service factory: {e}, using provider-specific fallback")
    
    # Provider-specific fallback
    provider = model_config.provider.value.lower() if hasattr(model_config, 'provider') else str(model_config).lower()
    model_name = model_config.model_name if hasattr(model_config, 'model_name') else "llama-3.3-70b-versatile"
    
    if provider == "groq":
        return run_groq_inference(prompts, base_prompt, model_name)
    else:
        # For non-groq providers, we need async context, so fallback to groq for now
        print(f"Provider {provider} not supported in sync context, falling back to groq")
        return run_groq_inference(prompts, base_prompt, "llama-3.3-70b-versatile")


def run_groq_inference(prompts: List[str], base_prompt: str, model: str = "llama-3.3-70b-versatile") -> List[str]:
    """
    Run inference using Groq LLaMA model with service or fallback to direct API calls.
    
    Args:
        prompts: List of input prompts
        base_prompt: Base instruction prompt to use as system prompt
        model: Model name to use
        
    Returns:
        List of model responses
    """
    if GROQ_SERVICE_AVAILABLE:
        # Use the GroqService for API calls
        try:
            import asyncio
            
            # Always run in new event loop for sync function
            async def _async_call():
                try:
                    groq_service = get_groq_service()
                    return await groq_service.batch_completions(
                        prompts=prompts,
                        base_prompt=base_prompt,
                        model_name=model,
                        component="metric_evaluator",
                        operation="batch_inference",
                        temperature=1.0,
                        max_completion_tokens=1024
                    )
                except Exception as e:
                    print(f"Error in groq service call: {e}")
                    raise e
            
            # Try to run in new event loop
            try:
                return asyncio.run(_async_call())
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    print("Already in event loop, falling back to direct API calls")
                else:
                    raise e
                
        except Exception as e:
            print(f"Error using Groq service, falling back to direct API: {e}")
            # Fall back to direct API calls
    
    # Fallback to direct API calls
    print("Using direct Groq API calls (service not available)")
    
    # Groq API Key
    groq_api_key = os.getenv("groq_api_key")
    
    responses = []
    for prompt in prompts:
        try:
            # Setup the API request payload
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_api_key}"
            }
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": base_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": model,
                "temperature": 1,
                "max_completion_tokens": 1024,
                "stream": False,
                "stop": None
            }
            
            # Make a simple API request
            response = httpx.post(url, json=payload, headers=headers, verify=False)
            response_data = response.json()
            
            # Log tokens to MongoDB (simple)
            try:
                from datetime import datetime
                client = MongoClient("mongodb://localhost:27017/")
                db = client["personal_project_log_usage"]
                collection = db["llm_usage"]
                
                usage = response_data.get("usage", {})
                log_entry = {
                    "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                    "provider": "groq",
                    "model": model,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/prompt_optimizer/core/metric.py",
                    "component": "metric_evaluator_fallback",
                    "operation": "direct_api_call"
                }
                collection.insert_one(log_entry)
            except Exception as e:
                print(f"Token logging failed: {e}")
            
            # Extract response content
            response_text = response_data["choices"][0]["message"]["content"]
            responses.append(response_text)
        except Exception as e:
            print(f"Error in Groq API call: {e}")
            responses.append("")  # Add empty string on error
    
    return responses


def extract_input_prompts(dataset_items, limit: int = 2) -> List[str]:
    """
    Extract input prompts from dataset items.
    
    Args:
        dataset_items: Items from langfuse dataset
        limit: Maximum number of prompts to extract
        
    Returns:
        List of input prompts for the model
    """
    prompts = []
    for i, item in enumerate(dataset_items):
        if i >= limit:  # Only process specified number of items
            break
            
        try:
            # Get the input text directly
            prompt = item.input
            prompts.append(prompt)
        except Exception as e:
            print(f"Error creating prompt: {e}")
            prompts.append("")  # Add empty string on error
    
    return prompts