from typing import Dict, Any, Optional
from fastapi import HTTPException, UploadFile
import uuid

from app.usecases.upload_dataset_usecase import UploadDatasetUseCase
from app.utils.error_handler import (
    OptimizationError,
    ValidationError,
    DatasetError
)
from app.utils.context_util import get_request_id

class UploadDatasetController:
    """
    Controller layer for handling dataset upload requests
    Manages HTTP-specific concerns and delegates to use cases
    """
    
    def __init__(self):
        self.upload_dataset_usecase = UploadDatasetUseCase()
    
    async def upload_csv_dataset(
        self, 
        file: UploadFile,
        dataset_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 3,
        retry_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Handle CSV dataset upload request
        
        Args:
            file: The uploaded CSV file
            dataset_name: Name for the dataset in LangFuse
            description: Optional description for the dataset
            metadata: Optional metadata for the dataset
            max_concurrent: Maximum concurrent uploads (default: 3, safe for free tier)
            retry_attempts: Number of retry attempts for rate limits (default: 3)
            
        Returns:
            Dict containing upload results
            
        Raises:
            HTTPException: On validation or processing errors
        """
        request_id = get_request_id()
        
        try:
            # Log the incoming request
            print(f"[{request_id}] Starting dataset upload request")
            print(f"[{request_id}] Dataset name: {dataset_name}")
            print(f"[{request_id}] File: {file.filename} ({file.content_type})")
            if description:
                print(f"[{request_id}] Description: {description[:100]}...")
            
            # Execute the upload through use case layer
            result = await self.upload_dataset_usecase.upload_csv_dataset(
                file=file,
                dataset_name=dataset_name,
                description=description,
                metadata=metadata,
                max_concurrent=max_concurrent,
                retry_attempts=retry_attempts
            )
            
            print(f"[{request_id}] Dataset upload completed successfully")
            print(f"[{request_id}] Total items uploaded: {result['upload_results']['total_items']}")
            print(f"[{request_id}] Successful: {result['upload_results']['successful']}")
            print(f"[{request_id}] Failed: {result['upload_results']['failed']}")
            
            return result
            
        except (ValidationError, DatasetError) as e:
            # These are user errors - return 400 status
            print(f"[{request_id}] Validation/Dataset error: {e.detail}")
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
            print(f"[{request_id}] Upload error: {e.detail}")
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
                    "message": "An unexpected error occurred during dataset upload",
                    "request_id": request_id
                }
            )
    
    async def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about an existing dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing dataset information
            
        Raises:
            HTTPException: If dataset not found or error occurs
        """
        request_id = get_request_id()
        
        try:
            print(f"[{request_id}] Getting dataset info for: {dataset_name}")
            
            result = await self.upload_dataset_usecase.get_dataset_info(dataset_name)
            
            print(f"[{request_id}] Dataset info retrieved successfully")
            
            return result
            
        except (ValidationError, DatasetError) as e:
            print(f"[{request_id}] Dataset error: {e.detail}")
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
            print(f"[{request_id}] Error getting dataset info: {e.detail}")
            raise HTTPException(
                status_code=e.status_code,
                detail={
                    "error": e.error_type,
                    "message": e.detail,
                    "request_id": request_id
                }
            )
        
        except Exception as e:
            print(f"[{request_id}] Unexpected error getting dataset info: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "Failed to get dataset information",
                    "request_id": request_id
                }
            )
    
    def validate_upload_request_format(self, request_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate the basic format of the upload request data
        
        Args:
            request_data: Raw request data
            
        Returns:
            Dict of validation errors (empty if valid)
        """
        errors = {}
        
        # Validate dataset_name
        if 'dataset_name' not in request_data:
            errors['dataset_name'] = "Field 'dataset_name' is required"
        elif not request_data['dataset_name']:
            errors['dataset_name'] = "Field 'dataset_name' cannot be empty"
        elif not isinstance(request_data['dataset_name'], str):
            errors['dataset_name'] = "Field 'dataset_name' must be a string"
        
        # Validate optional description
        if 'description' in request_data and request_data['description'] is not None:
            if not isinstance(request_data['description'], str):
                errors['description'] = "Field 'description' must be a string"
        
        # Validate optional metadata
        if 'metadata' in request_data and request_data['metadata'] is not None:
            if not isinstance(request_data['metadata'], dict):
                errors['metadata'] = "Field 'metadata' must be a dictionary"
        
        return errors 