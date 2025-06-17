from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

from app.services.optimization_service import OptimizationService
from app.models.optimization_models import (
    OptimizationRequest, 
    OptimizationResult, 
    OptimizationProgress
)
from app.utils.error_handler import (
    OptimizationError, 
    ValidationError,
    DatasetError,
    ModelConfigurationError,
    handle_optimization_exception
)
from app.utils.context_util import get_request_id

class OptimizationUseCase:
    """
    Use case layer for prompt optimization
    Handles business logic validation and coordinates with services
    """
    
    def __init__(self):
        self.optimization_service = OptimizationService()
    
    async def execute_optimization(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Execute the complete prompt optimization process
        
        Args:
            request: The optimization request containing all parameters
            
        Returns:
            OptimizationResult: Complete optimization results
            
        Raises:
            ValidationError: If request validation fails
            DatasetError: If dataset cannot be processed
            ModelConfigurationError: If model configuration is invalid
            OptimizationError: If optimization process fails
        """
        request_id = get_request_id()
        
        try:
            # Validate the optimization request
            await self._validate_optimization_request(request)
            
            # Execute the optimization
            result = await self.optimization_service.start_optimization(request)
            
            return result
            
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Handle unexpected errors
            optimization_error = handle_optimization_exception(
                e, request_id, "optimization use case"
            )
            raise optimization_error
    
    async def get_optimization_status(self, request_id: str) -> OptimizationProgress:
        """
        Get the current status of an optimization process
        
        Args:
            request_id: The optimization request ID
            
        Returns:
            OptimizationProgress: Current progress information
        """
        try:
            return await self.optimization_service.get_optimization_progress(request_id)
        except Exception as e:
            optimization_error = handle_optimization_exception(
                e, request_id, "status check"
            )
            raise optimization_error
    
    async def _validate_optimization_request(self, request: OptimizationRequest):
        """
        Validate the optimization request before processing
        
        Args:
            request: The optimization request to validate
            
        Raises:
            ValidationError: If validation fails
            DatasetError: If dataset validation fails
            ModelConfigurationError: If model configuration is invalid
        """
        request_id = get_request_id()
        
        # Validate prompts are not empty
        if not request.system_prompt.strip():
            raise ValidationError(
                "System prompt cannot be empty", 
                request_id=request_id, 
                field="system_prompt"
            )
        
        if not request.user_prompt.strip():
            raise ValidationError(
                "User prompt cannot be empty", 
                request_id=request_id, 
                field="user_prompt"
            )
        
        # Validate schema
        await self._validate_schema(request.json_schema, request_id)
        
        # Validate model configuration
        await self._validate_model_configuration(request.model_configuration, request_id)
        
        # Validate dataset
        await self._validate_dataset(request.dataset, request_id)
        
        # Validate optimization parameters
        await self._validate_optimization_parameters(request, request_id)
    
    async def _validate_schema(self, schema: Dict[str, Any], request_id: str):
        """Validate the JSON schema (supports nested structures)"""
        if not schema:
            raise ValidationError(
                "Schema cannot be empty", 
                request_id=request_id, 
                field="schema"
            )
        
        def validate_enum_values(field_name, enum_values, parent_path=""):
            """Recursively validate enum values (handles nested structures)"""
            full_path = f"{parent_path}.{field_name}" if parent_path else field_name
            
            if isinstance(enum_values, list):
                # Simple list of enum values
                if len(enum_values) == 0:
                    raise ValidationError(
                        f"Schema field '{full_path}' must have at least one enum value",
                        request_id=request_id,
                        field="schema"
                    )
                
                # Check for duplicate enum values
                if len(enum_values) != len(set(enum_values)):
                    raise ValidationError(
                        f"Schema field '{full_path}' has duplicate enum values",
                        request_id=request_id,
                        field="schema"
                    )
                
                # Validate each enum value
                for enum_val in enum_values:
                    if not isinstance(enum_val, str) or not enum_val.strip():
                        raise ValidationError(
                            f"Schema field '{full_path}' contains invalid enum value: {enum_val}",
                            request_id=request_id,
                            field="schema"
                        )
            elif isinstance(enum_values, dict):
                # Nested structure - validate each sub-field
                if len(enum_values) == 0:
                    raise ValidationError(
                        f"Schema field '{full_path}' cannot be empty dictionary",
                        request_id=request_id,
                        field="schema"
                    )
                for sub_field, sub_values in enum_values.items():
                    if not isinstance(sub_field, str) or not sub_field.strip():
                        raise ValidationError(
                            f"Schema field '{full_path}' contains invalid sub-field name: {sub_field}",
                            request_id=request_id,
                            field="schema"
                        )
                    validate_enum_values(sub_field, sub_values, full_path)
            else:
                raise ValidationError(
                    f"Schema field '{full_path}' must be either a list of strings or a dictionary of lists",
                    request_id=request_id,
                    field="schema"
                )
        
        for field_name, enum_values in schema.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValidationError(
                    f"Schema field name cannot be empty: {field_name}",
                    request_id=request_id,
                    field="schema"
                )
            validate_enum_values(field_name, enum_values)
    
    async def _validate_model_configuration(self, model_config, request_id: str):
        """Validate model configuration"""
        
        # Validate provider-specific requirements
        if model_config.provider.value == "openai":
            valid_openai_models = ["gpt-4.1-mini-2025-04-14"] 
            if model_config.model_name not in valid_openai_models:
                raise ModelConfigurationError(
                    f"Invalid OpenAI model name: {model_config.model_name}. Valid options: {valid_openai_models}",
                    request_id=request_id,
                    provider="openai"
                )
        
        elif model_config.provider.value == "anthropic":
            valid_anthropic_models = ["claude-sonnet-4-20250514"]
            if model_config.model_name not in valid_anthropic_models:
                raise ModelConfigurationError(
                    f"Invalid Anthropic model name: {model_config.model_name}. Valid options: {valid_anthropic_models}",
                    request_id=request_id,
                    provider="anthropic"
                )
        
        elif model_config.provider.value == "groq":
            valid_groq_models = ["llama-3.3-70b-versatile"]
            if model_config.model_name not in valid_groq_models:
                raise ModelConfigurationError(
                    f"Invalid Groq model name: {model_config.model_name}. Valid options: {valid_groq_models}",
                    request_id=request_id,
                    provider="groq"
                )
        
        elif model_config.provider.value == "google":
            valid_google_models = ["gemini-2.5-pro-preview-06-05"]
            if model_config.model_name not in valid_google_models:
                raise ModelConfigurationError(
                    f"Invalid Google model name: {model_config.model_name}. Valid options: {valid_google_models}",
                    request_id=request_id,
                    provider="google"
                )
        
        # Validate temperature range
        if model_config.temperature is not None:
            if not (0.0 <= model_config.temperature <= 2.0):
                raise ModelConfigurationError(
                    f"Temperature must be between 0.0 and 2.0, got {model_config.temperature}",
                    request_id=request_id,
                    provider=model_config.provider.value
                )
        
        # Validate max_tokens
        if model_config.max_tokens is not None:
            if model_config.max_tokens <= 0:
                raise ModelConfigurationError(
                    f"Max tokens must be positive, got {model_config.max_tokens}",
                    request_id=request_id,
                    provider=model_config.provider.value
                )
    
    async def _validate_dataset(self, dataset: str, request_id: str):
        """Validate dataset availability"""
        if not dataset.strip():
            raise DatasetError(
                "Dataset name cannot be empty",
                request_id=request_id
            )
        
        # Additional dataset validation could be added here
        # For example, checking if the dataset exists in LangFuse
        # or validating dataset format
    
    async def _validate_optimization_parameters(self, request: OptimizationRequest, request_id: str):
        """Validate optimization-specific parameters"""
        
        # Validate max_iterations
        if request.max_iterations is not None:
            if request.max_iterations < 1 or request.max_iterations > 10:
                raise ValidationError(
                    f"Max iterations must be between 1 and 10, got {request.max_iterations}",
                    request_id=request_id,
                    field="max_iterations"
                )
        
        # Validate improvement_threshold
        if request.improvement_threshold is not None:
            if request.improvement_threshold < 0.01 or request.improvement_threshold > 0.2:
                raise ValidationError(
                    f"Improvement threshold must be between 0.01 and 0.2, got {request.improvement_threshold}",
                    request_id=request_id,
                    field="improvement_threshold"
                )
    
    async def get_optimization_results(self, request_id: str) -> Dict[str, Any]:
        """
        Get complete optimization results from saved files
        
        Args:
            request_id: The optimization request ID
            
        Returns:
            Dict containing complete optimization results
            
        Raises:
            OptimizationError: If results not found
        """
        try:
            return await self.optimization_service.get_optimization_results_from_files(request_id)
        except Exception as e:
            optimization_error = handle_optimization_exception(
                e, request_id, "results retrieval"
            )
            raise optimization_error
    
    async def cleanup_optimization(self, request_id: str):
        """Clean up optimization resources"""
        try:
            await self.optimization_service.cleanup_optimization(request_id)
        except Exception as e:
            # Log cleanup errors but don't raise them
            print(f"Warning: Failed to cleanup optimization {request_id}: {e}") 