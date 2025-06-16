from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import uuid
import json

from app.controllers.upload_dataset_controller import UploadDatasetController
from app.utils.context_util import set_request_id, get_request_id

router = APIRouter(tags=["dataset"])

# Initialize controller
upload_dataset_controller = UploadDatasetController()

@router.post(
    "/upload-dataset",
    summary="Upload CSV Dataset to LangFuse",
    description="""
    Upload a CSV dataset to LangFuse for use in prompt optimization.
    
    The CSV file should have the following structure:
    - `input`: The input text/prompt for each example
    - `expected_output`: The expected JSON output (can be a JSON string or actual JSON)
    - Additional columns will be added as metadata
    
    **File Requirements:**
    - Must be a CSV file (.csv extension)
    - UTF-8 encoded
    - Maximum size: 10MB
    - Maximum items: 10,000
    
    **Dataset Name Requirements:**
    - At least 3 characters long
    - No special characters: / \\ : * ? " < > |
    
    **Example CSV Structure:**
    ```csv
    input,expected_output,category
    "Classify this text","{"sentiment": "positive", "intent": "feedback"}","example"
    "Another text","{"sentiment": "negative", "intent": "complaint"}","example"
    ```
    """
)
async def upload_csv_dataset(
    file: UploadFile = File(..., description="CSV file to upload"),
    dataset_name: str = Form(..., description="Name for the dataset in LangFuse"),
    description: Optional[str] = Form(None, description="Optional description for the dataset"),
    metadata: Optional[str] = Form(None, description="Optional metadata as JSON string"),
    max_concurrent: int = Form(3, description="Maximum concurrent uploads (1-20, default: 3)"),
    retry_attempts: int = Form(3, description="Number of retry attempts for rate limits (1-10, default: 3)")
) -> Dict[str, Any]:
    """
    Upload a CSV dataset to LangFuse
    
    **Form Data:**
    - `file`: CSV file (required)
    - `dataset_name`: Dataset name (required)
    - `description`: Dataset description (optional)
    - `metadata`: JSON string with additional metadata (optional)
    - `max_concurrent`: Maximum concurrent uploads (1-20, default: 3)
    - `retry_attempts`: Number of retry attempts for rate limits (1-10, default: 3)
    
    **Response includes:**
    - Dataset creation information
    - Upload statistics (successful/failed items)
    - File information
    - Request tracking ID
    
    **Performance tuning based on LangFuse API limits:**
    - **Free tier** (100 req/min ≈ 1.67/sec): Use max_concurrent=1-2
    - **Paid tier** (1000 req/min ≈ 16.67/sec): Use max_concurrent=3-10
    - Increase `retry_attempts` for better reliability with unstable connections
    - System automatically handles rate limits with exponential backoff
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    
    try:
        # Validate performance parameters
        if max_concurrent < 1 or max_concurrent > 20:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "max_concurrent must be between 1 and 20",
                    "request_id": request_id
                }
            )
        
        if retry_attempts < 1 or retry_attempts > 10:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "retry_attempts must be between 1 and 10",
                    "request_id": request_id
                }
            )
        
        # Parse metadata if provided
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                if not isinstance(parsed_metadata, dict):
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "validation_error",
                            "message": "Metadata must be a valid JSON object",
                            "request_id": request_id
                        }
                    )
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "validation_error",
                        "message": "Metadata must be valid JSON",
                        "request_id": request_id
                    }
                )
        
        # Execute upload
        result = await upload_dataset_controller.upload_csv_dataset(
            file=file,
            dataset_name=dataset_name,
            description=description,
            metadata=parsed_metadata,
            max_concurrent=max_concurrent,
            retry_attempts=retry_attempts
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
    "/datasets",
    summary="Get All Datasets from LangFuse",
    description="""
    Retrieve all datasets available in LangFuse for the current project.
    
    This endpoint fetches datasets directly from LangFuse, not just session uploads.
    Useful for selecting existing datasets for prompt optimization.
    
    **Query Parameters:**
    - `page`: Page number for pagination (default: 1)
    - `limit`: Number of datasets per page (default: 50, max: 100)
    
    **Response includes:**
    - List of available datasets with metadata
    - Pagination information
    - Total count and pages
    """
)
async def get_all_datasets(
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    limit: int = Query(50, ge=1, le=100, description="Number of datasets per page (max 100)")
) -> Dict[str, Any]:
    """
    Get all datasets from LangFuse
    
    **Query Parameters:**
    - `page`: Page number for pagination (default: 1)
    - `limit`: Number of datasets per page (default: 50, max: 100)
    
    **Response format:**
    ```json
    {
      "datasets": [
        {
          "id": "dataset_id",
          "name": "dataset_name", 
          "description": "Optional description",
          "metadata": {...},
          "createdAt": "2024-01-15T10:30:00Z",
          "updatedAt": "2024-01-15T10:30:00Z"
        }
      ],
      "pagination": {
        "page": 1,
        "limit": 50,
        "total_items": 150,
        "total_pages": 3,
        "has_next_page": true
      },
      "request_id": "uuid-here",
      "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    
    try:
        result = await upload_dataset_controller.get_all_datasets(page=page, limit=limit)
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
    "/dataset-upload-status",
    summary="Health check for dataset upload service",
    description="Returns service status and configuration information"
)
async def dataset_upload_health_check() -> Dict[str, Any]:
    """
    Health check for dataset upload service
    
    Returns service status and configuration information.
    """
    return {
        "status": "healthy",
        "service": "Dataset Upload Service",
        "version": "1.0.0",
        "features": {
            "csv_upload": True,
            "langfuse_integration": True,
            "validation": True,
            "metadata_support": True
        },
        "limits": {
            "max_file_size_mb": 10,
            "max_items_per_dataset": 10000,
            "supported_formats": ["CSV"]
        },
        "langfuse_connection": "configured"
    } 