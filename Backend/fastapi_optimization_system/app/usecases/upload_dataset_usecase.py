from typing import Dict, Any, Optional
from fastapi import UploadFile
import asyncio
from datetime import datetime

from app.services.langfuse_service import LangFuseService
from app.utils.error_handler import (
    DatasetError,
    ValidationError,
    handle_optimization_exception
)
from app.utils.context_util import get_request_id

class UploadDatasetUseCase:
    """
    Use case layer for dataset upload operations
    Handles business logic validation and coordinates with services
    """
    
    def __init__(self):
        self.langfuse_service = LangFuseService()
    
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
        Upload a CSV dataset to LangFuse
        
        Args:
            file: The uploaded CSV file
            dataset_name: Name for the dataset in LangFuse
            description: Optional description for the dataset
            metadata: Optional metadata for the dataset
            max_concurrent: Maximum concurrent uploads (default: 3, safe for free tier)
            retry_attempts: Number of retry attempts for rate limits (default: 3)
            
        Returns:
            Dict containing upload results and statistics
            
        Raises:
            ValidationError: If validation fails
            DatasetError: If dataset operations fail
        """
        request_id = get_request_id()
        
        try:
            # Validate the upload request
            await self._validate_upload_request(file, dataset_name)
            
            # Read and validate CSV file
            file_content = await self._read_and_validate_csv(file)
            
            # Parse CSV into dataset items
            dataset_items = self.langfuse_service.parse_csv_file(file_content)
            
            if not dataset_items:
                raise DatasetError(
                    "No valid dataset items found in CSV file",
                    request_id=request_id,
                    dataset_name=dataset_name
                )
            
            # Validate dataset items structure
            await self._validate_dataset_items(dataset_items, dataset_name)
            
            # Create dataset in LangFuse
            dataset_info = await self.langfuse_service.create_dataset(
                dataset_name=dataset_name,
                description=description,
                metadata=metadata
            )
            
            # Upload dataset items in batch
            upload_results = await self.langfuse_service.upload_dataset_items_batch(
                dataset_name=dataset_name,
                items=dataset_items,
                max_concurrent=max_concurrent,
                retry_attempts=retry_attempts
            )
            
            # Prepare response
            response = {
                "dataset_name": dataset_name,
                "dataset_info": dataset_info,
                "upload_results": upload_results,
                "file_info": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size_bytes": len(file_content)
                },
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
            
            return response
            
        except (ValidationError, DatasetError):
            # Re-raise validation and dataset errors as-is
            raise
        except Exception as e:
            # Handle unexpected errors
            optimization_error = handle_optimization_exception(
                e, request_id, "dataset upload"
            )
            raise optimization_error
    
    async def _validate_upload_request(self, file: UploadFile, dataset_name: str):
        """
        Validate the upload request parameters
        
        Args:
            file: The uploaded file
            dataset_name: Name for the dataset
            
        Raises:
            ValidationError: If validation fails
        """
        request_id = get_request_id()
        
        # Validate file
        if not file:
            raise ValidationError(
                "No file provided for upload",
                request_id=request_id,
                field="file"
            )
        
        if not file.filename:
            raise ValidationError(
                "File must have a filename",
                request_id=request_id,
                field="file"
            )
        
        # Validate file extension
        if not file.filename.lower().endswith('.csv'):
            raise ValidationError(
                "File must be a CSV file (.csv extension required)",
                request_id=request_id,
                field="file"
            )
        
        # Validate content type
        if file.content_type and not file.content_type.startswith('text/'):
            raise ValidationError(
                f"Invalid content type: {file.content_type}. Expected text/csv or similar",
                request_id=request_id,
                field="file"
            )
        
        # Validate dataset name
        if not dataset_name or not dataset_name.strip():
            raise ValidationError(
                "Dataset name cannot be empty",
                request_id=request_id,
                field="dataset_name"
            )
        
        # Validate dataset name format
        if len(dataset_name.strip()) < 3:
            raise ValidationError(
                "Dataset name must be at least 3 characters long",
                request_id=request_id,
                field="dataset_name"
            )
        
        # Check for invalid characters in dataset name
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in dataset_name for char in invalid_chars):
            raise ValidationError(
                f"Dataset name contains invalid characters: {invalid_chars}",
                request_id=request_id,
                field="dataset_name"
            )
    
    async def _read_and_validate_csv(self, file: UploadFile) -> bytes:
        """
        Read and validate the CSV file content
        
        Args:
            file: The uploaded CSV file
            
        Returns:
            bytes: The file content
            
        Raises:
            ValidationError: If file reading or validation fails
        """
        request_id = get_request_id()
        
        try:
            # Read file content
            file_content = await file.read()
            
            # Validate file size (max 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                raise ValidationError(
                    f"File size ({len(file_content)} bytes) exceeds maximum allowed size ({max_size} bytes)",
                    request_id=request_id,
                    field="file"
                )
            
            # Validate file is not empty
            if len(file_content) == 0:
                raise ValidationError(
                    "File is empty",
                    request_id=request_id,
                    field="file"
                )
            
            # Try to decode as UTF-8
            try:
                content_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                raise ValidationError(
                    "File must be UTF-8 encoded",
                    request_id=request_id,
                    field="file"
                )
            
            # Basic CSV validation - check if it has at least one comma or newline
            if ',' not in content_str and '\n' not in content_str:
                raise ValidationError(
                    "File does not appear to be a valid CSV format",
                    request_id=request_id,
                    field="file"
                )
            
            return file_content
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Error reading file: {str(e)}",
                request_id=request_id,
                field="file"
            )
    
    async def _validate_dataset_items(self, dataset_items: list, dataset_name: str):
        """
        Validate the parsed dataset items
        
        Args:
            dataset_items: List of parsed dataset items
            dataset_name: Name of the dataset
            
        Raises:
            DatasetError: If validation fails
        """
        request_id = get_request_id()
        
        if not dataset_items:
            raise DatasetError(
                "No dataset items found after parsing CSV",
                request_id=request_id,
                dataset_name=dataset_name
            )
        
        # Validate minimum number of items
        if len(dataset_items) < 1:
            raise DatasetError(
                "Dataset must contain at least 1 item",
                request_id=request_id,
                dataset_name=dataset_name
            )
        
        # Validate maximum number of items (to prevent overwhelming LangFuse)
        max_items = 10000  # Reasonable limit
        if len(dataset_items) > max_items:
            raise DatasetError(
                f"Dataset contains too many items ({len(dataset_items)}). Maximum allowed: {max_items}",
                request_id=request_id,
                dataset_name=dataset_name
            )
        
        # Validate structure of first few items
        sample_size = min(5, len(dataset_items))
        for i, item in enumerate(dataset_items[:sample_size]):
            if not isinstance(item, dict):
                raise DatasetError(
                    f"Dataset item {i+1} is not a valid dictionary",
                    request_id=request_id,
                    dataset_name=dataset_name
                )
            
            # Check required fields
            if 'input' not in item:
                raise DatasetError(
                    f"Dataset item {i+1} missing required 'input' field",
                    request_id=request_id,
                    dataset_name=dataset_name
                )
            
            if 'expected_output' not in item:
                raise DatasetError(
                    f"Dataset item {i+1} missing required 'expected_output' field",
                    request_id=request_id,
                    dataset_name=dataset_name
                )
            
            # Validate input is not empty
            if not item['input'] or not str(item['input']).strip():
                raise DatasetError(
                    f"Dataset item {i+1} has empty 'input' field",
                    request_id=request_id,
                    dataset_name=dataset_name
                )
    
    async def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about an existing dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing dataset information
        """
        request_id = get_request_id()
        
        try:
            if not dataset_name or not dataset_name.strip():
                raise ValidationError(
                    "Dataset name cannot be empty",
                    request_id=request_id,
                    field="dataset_name"
                )
            
            dataset_info = await self.langfuse_service.get_dataset(dataset_name)
            
            return {
                "dataset_info": dataset_info,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            optimization_error = handle_optimization_exception(
                e, request_id, "get dataset info"
            )
            raise optimization_error
