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
    description="Get the current status and progress of an optimization request"
)
async def get_optimization_status(
    request_id: str
) -> OptimizationProgress:
    """
    Get optimization progress status
    
    Returns real-time progress information including:
    - Current processing step
    - Progress percentage
    - Current iteration (if in optimization loop)
    - Estimated time remaining
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
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "description": "OpenAI GPT models"
            },
            "anthropic": {
                "models": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
                "description": "Anthropic Claude models"
            },
            "groq": {
                "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"],
                "description": "Groq LLaMA models"
            },
            "google": {
                "models": ["gemini-pro", "gemini-pro-vision"],
                "description": "Google Gemini models"
            }
        },
        "recommendations": {
            "development": "groq/llama-3.3-70b-versatile",
            "production": "anthropic/claude-3-sonnet",
            "cost_effective": "openai/gpt-3.5-turbo"
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
                        "model_name": "gpt-4",
                        "temperature": 0.1
                    },
                    "dataset": "ecommerce_products",
                    "max_iterations": 3,
                    "improvement_threshold": 0.03,
                    "enable_human_feedback": True
                }
            }
        }
    } 