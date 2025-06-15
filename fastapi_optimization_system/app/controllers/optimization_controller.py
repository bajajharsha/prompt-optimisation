from typing import Dict, Any
from fastapi import HTTPException
import asyncio

from app.usecases.optimization_usecase import OptimizationUseCase
from app.models.optimization_models import (
    OptimizationRequest, 
    OptimizationResult, 
    OptimizationProgress
)
from app.utils.error_handler import (
    OptimizationError,
    ValidationError,
    DataProcessingError,
    OptimizationProcessError,
    ModelConfigurationError,
    DatasetError
)
from app.utils.context_util import get_request_id

class OptimizationController:
    """
    Controller layer for handling optimization requests
    Manages HTTP-specific concerns and delegates to use cases
    """
    
    def __init__(self):
        self.optimization_usecase = OptimizationUseCase()
    
    async def optimize_prompt(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Handle prompt optimization request
        
        Args:
            request: The optimization request
            
        Returns:
            OptimizationResult: The optimization results
            
        Raises:
            HTTPException: On validation or processing errors
        """
        request_id = get_request_id()
        
        try:
            # Log the incoming request
            print(f"[{request_id}] Starting optimization request")
            print(f"[{request_id}] Dataset: {request.dataset}")
            print(f"[{request_id}] Model: {request.model_configuration.provider.value}/{request.model_configuration.model_name}")
            print(f"[{request_id}] Max iterations: {request.max_iterations}")
            
            # Execute the optimization through use case layer
            result = await self.optimization_usecase.execute_optimization(request)
            
            print(f"[{request_id}] Optimization completed successfully")
            print(f"[{request_id}] Total iterations: {result.total_iterations}")
            print(f"[{request_id}] Improvement: {result.improvement_percentage:.2f}%")
            print(f"[{request_id}] Recommendation: {result.deployment_recommendation}")
            
            return result
            
        except (ValidationError, DatasetError, ModelConfigurationError) as e:
            # These are user errors - return 400 status
            print(f"[{request_id}] Validation error: {e.detail}")
            raise HTTPException(
                status_code=e.status_code,
                detail={
                    "error": e.error_type,
                    "message": e.detail,
                    "request_id": request_id,
                    "details": e.additional_details
                }
            )
        
        except (DataProcessingError, OptimizationProcessError) as e:
            # These are server errors - return 500 status
            print(f"[{request_id}] Processing error: {e.detail}")
            raise HTTPException(
                status_code=e.status_code,
                detail={
                    "error": e.error_type,
                    "message": e.detail,
                    "request_id": request_id,
                    "details": e.additional_details
                }
            )
        
        except OptimizationError as e:
            # Generic optimization errors
            print(f"[{request_id}] Optimization error: {e.detail}")
            raise HTTPException(
                status_code=e.status_code,
                detail={
                    "error": e.error_type,
                    "message": e.detail,
                    "request_id": request_id,
                    "details": e.additional_details
                }
            )
        
        except Exception as e:
            # Unexpected errors
            print(f"[{request_id}] Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "An unexpected error occurred during optimization",
                    "request_id": request_id
                }
            )
    
    async def get_optimization_status(self, request_id: str) -> OptimizationProgress:
        """
        Get optimization progress status
        
        Args:
            request_id: The optimization request ID
            
        Returns:
            OptimizationProgress: Current progress information
            
        Raises:
            HTTPException: If request not found or error occurs
        """
        try:
            progress = await self.optimization_usecase.get_optimization_status(request_id)
            return progress
            
        except OptimizationError as e:
            if e.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "not_found",
                        "message": f"Optimization request {request_id} not found",
                        "request_id": request_id
                    }
                )
            else:
                raise HTTPException(
                    status_code=e.status_code,
                    detail={
                        "error": e.error_type,
                        "message": e.detail,
                        "request_id": request_id
                    }
                )
        
        except Exception as e:
            print(f"[{request_id}] Error getting status: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "Failed to get optimization status",
                    "request_id": request_id
                }
            )
    
    async def cancel_optimization(self, request_id: str) -> Dict[str, Any]:
        """
        Cancel an ongoing optimization
        
        Args:
            request_id: The optimization request ID
            
        Returns:
            Dict containing cancellation confirmation
        """
        try:
            # Cleanup optimization resources
            await self.optimization_usecase.cleanup_optimization(request_id)
            
            return {
                "message": "Optimization cancelled successfully",
                "request_id": request_id,
                "status": "cancelled"
            }
            
        except Exception as e:
            print(f"[{request_id}] Error cancelling optimization: {str(e)}")
            # Don't raise error for cleanup failures
            return {
                "message": "Optimization cancellation completed with warnings",
                "request_id": request_id,
                "status": "cancelled",
                "warning": str(e)
            }
    
    def validate_request_format(self, request_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate the basic format of the request data
        
        Args:
            request_data: Raw request data
            
        Returns:
            Dict of validation errors (empty if valid)
        """
        errors = {}
        
        required_fields = ['system_prompt', 'user_prompt', 'schema', 'model_configuration', 'dataset']
        
        for field in required_fields:
            if field not in request_data:
                errors[field] = f"Field '{field}' is required"
            elif not request_data[field]:
                errors[field] = f"Field '{field}' cannot be empty"
        
        # Validate model_configuration structure
        if 'model_configuration' in request_data and isinstance(request_data['model_configuration'], dict):
            model_config = request_data['model_configuration']
            model_required_fields = ['provider', 'model_name']
            
            for field in model_required_fields:
                if field not in model_config:
                    errors[f'model_configuration.{field}'] = f"Model config field '{field}' is required"
        
        # Validate schema structure
        if 'schema' in request_data:
            schema = request_data['schema']
            if not isinstance(schema, dict):
                errors['schema'] = "Schema must be a dictionary"
            elif len(schema) == 0:
                errors['schema'] = "Schema cannot be empty"
        
        return errors 