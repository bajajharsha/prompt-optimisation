import traceback
from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional

class OptimizationError(Exception):
    """Custom exception for optimization-related errors"""
    
    def __init__(
        self, 
        detail: str, 
        status_code: int = 400,
        error_type: str = "optimization_error",
        request_id: Optional[str] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_type = error_type
        self.request_id = request_id
        self.additional_details = additional_details or {}
        self.traceback = traceback.format_exc()
        super().__init__(self.detail)
    
    @property
    def response(self) -> JSONResponse:
        """Generate JSON response for this error"""
        return JSONResponse(
            status_code=self.status_code,
            content={
                "error": self.error_type,
                "message": self.detail,
                "request_id": self.request_id,
                "details": self.additional_details,
                "timestamp": datetime.now().isoformat()
            }
        )

class ValidationError(OptimizationError):
    """Validation-specific error"""
    def __init__(self, detail: str, request_id: Optional[str] = None, field: Optional[str] = None):
        additional_details = {"field": field} if field else {}
        super().__init__(
            detail=detail,
            status_code=422,
            error_type="validation_error",
            request_id=request_id,
            additional_details=additional_details
        )

class DataProcessingError(OptimizationError):
    """Data processing error"""
    def __init__(self, detail: str, request_id: Optional[str] = None, step: Optional[str] = None):
        additional_details = {"processing_step": step} if step else {}
        super().__init__(
            detail=detail,
            status_code=500,
            error_type="data_processing_error",
            request_id=request_id,
            additional_details=additional_details
        )

class OptimizationProcessError(OptimizationError):
    """Optimization process error"""
    def __init__(self, detail: str, request_id: Optional[str] = None, iteration: Optional[int] = None):
        additional_details = {"iteration": iteration} if iteration is not None else {}
        super().__init__(
            detail=detail,
            status_code=500,
            error_type="optimization_process_error",
            request_id=request_id,
            additional_details=additional_details
        )

class ModelConfigurationError(OptimizationError):
    """Model configuration error"""
    def __init__(self, detail: str, request_id: Optional[str] = None, provider: Optional[str] = None):
        additional_details = {"provider": provider} if provider else {}
        super().__init__(
            detail=detail,
            status_code=400,
            error_type="model_configuration_error",
            request_id=request_id,
            additional_details=additional_details
        )

class DatasetError(OptimizationError):
    """Dataset-related error"""
    def __init__(self, detail: str, request_id: Optional[str] = None, dataset_name: Optional[str] = None):
        additional_details = {"dataset": dataset_name} if dataset_name else {}
        super().__init__(
            detail=detail,
            status_code=404,
            error_type="dataset_error",
            request_id=request_id,
            additional_details=additional_details
        )

def handle_optimization_exception(
    exception: Exception, 
    request_id: Optional[str] = None,
    context: Optional[str] = None
) -> OptimizationError:
    """Convert generic exceptions to OptimizationError"""
    
    if isinstance(exception, OptimizationError):
        return exception
    
    detail = f"Unexpected error during {context}: {str(exception)}" if context else str(exception)
    
    return OptimizationError(
        detail=detail,
        status_code=500,
        error_type="internal_error",
        request_id=request_id,
        additional_details={"original_exception": type(exception).__name__}
    ) 