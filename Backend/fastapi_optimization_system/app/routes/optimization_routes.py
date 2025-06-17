from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import uuid

from app.controllers.optimization_controller import OptimizationController
from app.models.optimization_models import (
    OptimizationRequest, 
    OptimizationResult, 
    OptimizationProgress
)
from app.utils.context_util import set_request_id, get_request_id

router = APIRouter(tags=["optimization"])

# Initialize controller
optimization_controller = OptimizationController()

@router.post(
    "/optimize",
    response_model=OptimizationResult,
    summary="Execute Prompt Optimization",
    description="""
    Execute the complete prompt optimization process for JSON responses with enum validation.
    
    This endpoint:
    1. Splits the dataset into train/dev_a/dev_b/test
    2. Evaluates baseline performance 
    3. Runs optimization iterations
    4. Collects human feedback via LangFuse
    5. Returns the best optimized prompt with detailed metrics
    
    The process is fully automated and may take several minutes to complete.
    """
)
async def optimize_prompt(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
) -> OptimizationResult:
    """
    Execute prompt optimization
    
    **Request Body Example:**
    ```json
    {
        "system_prompt": "You are a classifier...",
        "user_prompt": "Classify the following text...",
        "schema": {
            "intent": ["complaint", "feedback", "question"],
            "urgency": ["low", "medium", "high"]
        },
        "model_configuration": {
            "provider": "groq",
            "model_name": "llama-3.3-70b-versatile",
            "temperature": 0.2
        },
        "dataset": "classification_dataset",
        "max_iterations": 5,
        "improvement_threshold": 0.05,
        "enable_human_feedback": true
    }
    ```
    """
    # Generate request ID if not provided
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    
    try:
        # Execute optimization
        result = await optimization_controller.optimize_prompt(request)
        
        # Add cleanup task to background
        background_tasks.add_task(
            optimization_controller.optimization_usecase.cleanup_optimization,
            request_id
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions from controller
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Unexpected error: {str(e)}",
                "request_id": request_id
            }
        )

@router.get(
    "/optimize/{request_id}/status",
    response_model=OptimizationProgress,
    summary="Get Optimization Status",
    description="Get the current status and progress of an optimization request from saved files"
)
async def get_optimization_status(
    request_id: str
) -> OptimizationProgress:
    """
    Get optimization progress status from saved intermediate results
    
    Returns real-time progress information including:
    - Current processing step
    - Progress percentage
    - Current iteration (if in optimization loop)
    - Estimated time remaining
    
    This endpoint reads directly from saved files, making it persistent across server restarts.
    """
    try:
        return await optimization_controller.get_optimization_status(request_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Failed to get status: {str(e)}",
                "request_id": request_id
            }
        )

@router.get(
    "/optimizations",
    summary="List All Optimization Results",
    description="Get a list of all available optimization results from saved files"
)
async def list_optimization_results() -> Dict[str, Any]:
    """
    Get list of all available optimization results
    
    Returns a list of optimization results with basic metadata for each,
    useful for selecting specific optimizations to view in detail.
    """
    try:
        import os
        from datetime import datetime
        
        results_base_dir = "intermediate_results"
        
        if not os.path.exists(results_base_dir):
            return {
                "optimizations": [],
                "total_count": 0,
                "message": "No optimization results found"
            }
        
        optimizations = []
        
        for item in os.listdir(results_base_dir):
            item_path = os.path.join(results_base_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory has results
                final_dir = os.path.join(item_path, "final")
                optimization_results_file = os.path.join(item_path, "optimization_results.json")
                
                optimization_info = {
                    "request_id": item,
                    "status": "unknown",
                    "created": "Unknown",
                    "last_modified": "Unknown",
                    "has_final_results": False,
                    "has_optimization_results": False,
                    "iterations": 0,
                    "improvement": 0.0,
                    "deployment_recommendation": "unknown"
                }
                
                # Get creation time
                try:
                    creation_time = os.path.getctime(item_path)
                    optimization_info["created"] = datetime.fromtimestamp(creation_time).isoformat()
                except:
                    pass
                
                # Get last modified time
                try:
                    modified_time = os.path.getmtime(item_path)
                    optimization_info["last_modified"] = datetime.fromtimestamp(modified_time).isoformat()
                except:
                    pass
                
                # Check for final results
                if os.path.exists(final_dir):
                    optimization_info["has_final_results"] = True
                    optimization_info["status"] = "completed"
                    
                    # Try to read final results for more details
                    try:
                        final_results_file = os.path.join(final_dir, "final_prompt.json")
                        if os.path.exists(final_results_file):
                            import json
                            with open(final_results_file, "r") as f:
                                final_data = json.load(f)
                                optimization_info["deployment_recommendation"] = final_data.get("deployment_decision", "unknown")
                                optimization_info["improvement"] = final_data.get("test_improvement", 0.0)
                    except:
                        pass
                
                # Check for optimization results
                if os.path.exists(optimization_results_file):
                    optimization_info["has_optimization_results"] = True
                    if optimization_info["status"] == "unknown":
                        optimization_info["status"] = "optimization_completed"
                    
                    # Try to read optimization results for iteration count and consistent improvement
                    try:
                        import json
                        with open(optimization_results_file, "r") as f:
                            opt_data = json.load(f)
                            optimization_info["iterations"] = opt_data.get("total_iterations", 0)
                            
                            # Calculate Dev A improvement for consistency with detailed results
                            # Try to get the same improvement calculation as the detailed results page
                            final_metrics = opt_data.get("final_metrics", {})
                            
                            # Check if we can read dev_a_baseline_metrics from same directory
                            dev_a_baseline_file = os.path.join(item_path, "dev_a_baseline_metrics.json")
                            if os.path.exists(dev_a_baseline_file) and final_metrics:
                                try:
                                    with open(dev_a_baseline_file, "r") as baseline_f:
                                        dev_a_baseline = json.load(baseline_f)
                                        
                                    # Calculate the same Dev A improvement as shown in detailed results
                                    baseline_acc = dev_a_baseline.get('overall_accuracy', 0)
                                    optimized_acc = final_metrics.get('overall_accuracy', 0)
                                    dev_a_improvement = optimized_acc - baseline_acc
                                    
                                    if dev_a_improvement != 0:
                                        optimization_info["improvement"] = dev_a_improvement
                                        optimization_info["improvement_type"] = "dev_a"
                                    else:
                                        # Fallback to best_candidate improvement
                                        if "best_candidate" in opt_data:
                                            best_candidate = opt_data["best_candidate"]
                                            optimization_info["improvement"] = best_candidate.get("improvement_over_baseline", 0.0)
                                            optimization_info["improvement_type"] = "best_candidate"
                                except:
                                    # Fallback to best_candidate improvement
                                    if "best_candidate" in opt_data:
                                        best_candidate = opt_data["best_candidate"]
                                        optimization_info["improvement"] = best_candidate.get("improvement_over_baseline", 0.0)
                                        optimization_info["improvement_type"] = "best_candidate"
                            else:
                                # Fallback to best_candidate improvement
                                if "best_candidate" in opt_data:
                                    best_candidate = opt_data["best_candidate"]
                                    optimization_info["improvement"] = best_candidate.get("improvement_over_baseline", 0.0)
                                    optimization_info["improvement_type"] = "best_candidate"
                    except:
                        pass
                
                # Count iteration directories for status
                if optimization_info["status"] == "unknown":
                    iteration_dirs = [d for d in os.listdir(item_path) if d.startswith("iteration_") and d.endswith("_selected")]
                    if iteration_dirs:
                        optimization_info["status"] = "in_progress"
                        optimization_info["iterations"] = len(iteration_dirs)
                    elif os.path.exists(os.path.join(item_path, "data_splits.json")):
                        optimization_info["status"] = "initialized"
                
                optimizations.append(optimization_info)
        
        # Sort by last modified time (newest first)
        optimizations.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return {
            "optimizations": optimizations,
            "total_count": len(optimizations),
            "message": f"Found {len(optimizations)} optimization results"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Failed to list optimization results: {str(e)}"
            }
        )

@router.get(
    "/optimize/{request_id}/results",
    summary="Get Complete Optimization Results",
    description="Get complete optimization results from saved files (persistent across server restarts)"
)
async def get_optimization_results(
    request_id: str
) -> Dict[str, Any]:
    """
    Get complete optimization results from saved intermediate files
    
    Returns comprehensive results including:
    - Data splits information
    - Baseline metrics
    - All iteration results
    - Selected prompts per iteration
    - Final recommendations
    - Human feedback summaries
    
    This endpoint reads directly from saved files, providing access to results
    even after server restarts.
    """
    try:
        return await optimization_controller.get_optimization_results(request_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Failed to get results: {str(e)}",
                "request_id": request_id
            }
        )

@router.delete(
    "/optimize/{request_id}",
    summary="Cancel Optimization",
    description="Cancel an ongoing optimization process and cleanup resources"
)
async def cancel_optimization(
    request_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Cancel an ongoing optimization
    
    This will:
    1. Stop the optimization process
    2. Cleanup allocated resources
    3. Return cancellation confirmation
    """
    try:
        result = await optimization_controller.cancel_optimization(request_id)
        return result
    except Exception as e:
        # Don't raise errors for cancellation - just return warning
        return {
            "message": "Cancellation completed with warnings",
            "request_id": request_id,
            "status": "cancelled",
            "warning": str(e)
        }

@router.post(
    "/validate",
    summary="Validate Optimization Request",
    description="Validate an optimization request without executing it"
)
async def validate_optimization_request(
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate optimization request format and parameters
    
    Useful for client-side validation before submitting the actual optimization request.
    """
    try:
        # Basic format validation
        format_errors = optimization_controller.validate_request_format(request_data)
        
        if format_errors:
            return {
                "valid": False,
                "errors": format_errors,
                "message": "Request format validation failed"
            }
        
        # Try to parse as Pydantic model for deeper validation
        try:
            OptimizationRequest(**request_data)
            return {
                "valid": True,
                "message": "Request validation passed"
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": {"validation": str(e)},
                "message": "Request validation failed"
            }
            
    except Exception as e:
        return {
            "valid": False,
            "errors": {"unexpected": str(e)},
            "message": "Validation process failed"
        }

@router.get(
    "/health",
    summary="Health Check",
    description="Check if the optimization service is healthy and ready to accept requests"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns service status and basic configuration information.
    """
    return {
        "status": "healthy",
        "service": "Auto Prompt Optimization API",
        "version": "1.0.0",
        "features": {
            "optimization": True,
            "human_feedback": True,
            "progress_tracking": True,
            "cancellation": True
        },
        "supported_providers": ["openai", "anthropic", "groq", "google"]
    }

@router.get(
    "/models",
    summary="List Supported Models",
    description="Get a list of supported model providers and their available models"
)
async def list_supported_models() -> Dict[str, Any]:
    """
    List supported models by provider
    
    Returns information about which models are supported for each provider.
    """
    return {
        "providers": {
            "openai": {
                "models": ["gpt-4.1-mini-2025-04-14"],
                "description": "OpenAI GPT models"
            },
            "anthropic": {
                "models": ["claude-sonnet-4-20250514"],
                "description": "Anthropic Claude models"
            },
            "groq": {
                "models": ["llama-3.3-70b-versatile"],
                "description": "Groq LLaMA models"
            },
            "google": {
                "models": ["gemini-2.5-pro-preview-06-05"],
                "description": "Google Gemini models"
            }
        },
        "recommendations": {
            "development": "groq/llama-3.3-70b-versatile",
            "production": "anthropic/claude-sonnet-4-20250514",
            "cost_effective": "openai/gpt-4.1-mini-2025-04-14"
        }
    }

@router.get(
    "/examples",
    summary="Get Request Examples",
    description="Get example requests for different use cases"
)
async def get_request_examples() -> Dict[str, Any]:
    """
    Get example optimization requests
    
    Returns sample requests for common use cases to help with API integration.
    """
    return {
        "examples": {
            "text_classification": {
                "description": "Basic text classification with sentiment and intent",
                "request": {
                    "system_prompt": "You are a text classifier. Classify the input text according to the provided schema.",
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
                    "max_iterations": 5,
                    "improvement_threshold": 0.05,
                    "enable_human_feedback": True
                }
            },
            "ecommerce_classification": {
                "description": "E-commerce product categorization",
                "request": {
                    "system_prompt": "You are an e-commerce product classifier. Categorize products based on their descriptions.",
                    "user_prompt": "Categorize this product description.",
                    "schema": {
                        "category": ["electronics", "clothing", "home", "books", "toys"],
                        "price_range": ["budget", "mid", "premium"],
                        "target_audience": ["kids", "adults", "seniors"]
                    },
                    "model_configuration": {
                        "provider": "openai",
                        "model_name": "gpt-4.1-mini-2025-04-14",
                        "temperature": 0.1
                    },
                    "dataset": "ecommerce_products",
                    "max_iterations": 3,
                    "improvement_threshold": 0.03,
                    "enable_human_feedback": True
                }
            },
            "nested_classification": {
                "description": "Complex nested classification with multiple levels",
                "request": {
                    "system_prompt": "You are an advanced classifier that analyzes user requests with multiple dimensions.",
                    "user_prompt": "Analyze this user input across all dimensions.",
                    "schema": {
                        "user": {
                            "profile": {
                                "age_group": ["young", "adult", "senior"],
                                "experience_level": ["beginner", "intermediate", "expert"]
                            },
                            "intent": ["question", "complaint", "request", "feedback"]
                        },
                        "content": {
                            "category": ["technical", "business", "personal"],
                            "urgency": ["low", "medium", "high", "critical"],
                            "complexity": ["simple", "moderate", "complex"]
                        }
                    },
                    "model_configuration": {
                        "provider": "anthropic",
                        "model_name": "claude-sonnet-4-20250514",
                        "temperature": 0.1
                    },
                    "dataset": "complex_classification_dataset",
                    "max_iterations": 4,
                    "improvement_threshold": 0.04,
                    "enable_human_feedback": True
                }
            }
        }
    } 