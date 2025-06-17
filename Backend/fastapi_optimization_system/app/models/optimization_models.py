from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Union
from enum import Enum

class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"

class ModelConfiguration(BaseModel):
    """Model configuration for optimization"""
    model_config = {"protected_namespaces": ()}
    
    provider: ModelProvider = Field(..., description="Model provider")
    model_name: str = Field(..., description="Specific model name")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

class OptimizationRequest(BaseModel):
    """Main request model for prompt optimization"""
    model_config = {"protected_namespaces": ()}
    
    system_prompt: str = Field(..., description="System prompt to optimize")
    user_prompt: str = Field(..., description="User prompt context")
    json_schema: Dict[str, Union[List[str], Dict[str, Union[List[str], Dict[str, List[str]]]]]] = Field(..., description="JSON schema with enum values (supports nested structures up to 3 levels)", alias="schema")
    model_configuration: ModelConfiguration = Field(..., description="Model configuration")
    dataset: str = Field(..., description="Dataset name or path")
    
    # Optional parameters
    max_iterations: Optional[int] = Field(5, ge=1, le=10, description="Maximum optimization iterations")
    improvement_threshold: Optional[float] = Field(0.05, ge=0.01, le=0.2, description="Minimum improvement threshold")
    enable_human_feedback: Optional[bool] = Field(True, description="Enable human feedback collection")
    
    @field_validator('system_prompt', 'user_prompt')
    @classmethod
    def validate_prompts(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompts cannot be empty")
        return v.strip()
    
    @field_validator('json_schema')
    @classmethod
    def validate_schema(cls, v):
        if not v:
            raise ValueError("Schema cannot be empty")
        
        def validate_enum_values(field_name, enum_values, parent_path=""):
            """Recursively validate enum values (handles nested structures)"""
            full_path = f"{parent_path}.{field_name}" if parent_path else field_name
            
            if isinstance(enum_values, list):
                # Simple list of enum values
                if len(enum_values) == 0:
                    raise ValueError(f"Schema field '{full_path}' must have at least one enum value")
                for enum_val in enum_values:
                    if not isinstance(enum_val, str) or not enum_val.strip():
                        raise ValueError(f"Schema field '{full_path}' contains invalid enum value: {enum_val}")
            elif isinstance(enum_values, dict):
                # Nested structure - validate each sub-field
                if len(enum_values) == 0:
                    raise ValueError(f"Schema field '{full_path}' cannot be empty dictionary")
                for sub_field, sub_values in enum_values.items():
                    if not isinstance(sub_field, str) or not sub_field.strip():
                        raise ValueError(f"Schema field '{full_path}' contains invalid sub-field name: {sub_field}")
                    validate_enum_values(sub_field, sub_values, full_path)
            else:
                raise ValueError(f"Schema field '{full_path}' must be either a list of strings or a dictionary of lists")
        
        for field_name, enum_values in v.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError(f"Schema field name cannot be empty: {field_name}")
            validate_enum_values(field_name, enum_values)
        
        return v

class OptimizationIterationResult(BaseModel):
    """Result of a single optimization iteration"""
    iteration: int = Field(..., description="Iteration number")
    optimizer_used: str = Field(..., description="Optimizer that generated this result")
    candidate_prompt: str = Field(..., description="Generated candidate prompt")
    dev_a_metrics: Dict[str, Any] = Field(..., description="Metrics on dev_a dataset")
    improvement_over_baseline: float = Field(..., description="Improvement percentage")
    reasoning: str = Field(..., description="Reasoning for the optimization")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

class HumanFeedbackSummary(BaseModel):
    """Summary of human feedback"""
    total_cases_reviewed: int = Field(..., description="Total cases reviewed by humans")
    accuracy_improvement: float = Field(..., description="Accuracy improvement based on feedback")
    key_insights: List[str] = Field(..., description="Key insights from human feedback")
    problematic_fields: List[str] = Field(..., description="Fields that need attention")

class DataSplitSummary(BaseModel):
    """Summary of data split"""
    total_samples: int = Field(..., description="Total samples in dataset")
    train_samples: int = Field(..., description="Training samples (25%)")
    dev_a_samples: int = Field(..., description="Dev A samples (35%)")
    dev_b_samples: int = Field(..., description="Dev B samples (20%)")
    test_samples: int = Field(..., description="Test samples (20%)")

class OptimizationResult(BaseModel):
    """Final optimization result"""
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Optimization status")
    
    # Data information
    data_split: DataSplitSummary = Field(..., description="Data split summary")
    
    # Baseline performance
    baseline_prompt: str = Field(..., description="Original baseline prompt")
    baseline_metrics: Dict[str, Any] = Field(..., description="Baseline performance metrics")
    
    # Optimization process
    total_iterations: int = Field(..., description="Total iterations performed")
    iterations_history: List[OptimizationIterationResult] = Field(..., description="History of all iterations")
    
    # Best result
    best_prompt: str = Field(..., description="Best optimized prompt")
    best_metrics: Dict[str, Any] = Field(..., description="Best performance metrics")
    improvement_percentage: float = Field(..., description="Overall improvement percentage")
    
    # Human feedback
    human_feedback_summary: Optional[HumanFeedbackSummary] = Field(None, description="Human feedback summary")
    
    # Final evaluation
    test_metrics: Dict[str, Any] = Field(..., description="Final test metrics")
    deployment_recommendation: str = Field(..., description="Deployment recommendation")
    
    # Metadata
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    timestamp: str = Field(..., description="Optimization completion timestamp")
    
class OptimizationProgress(BaseModel):
    """Progress update for long-running optimization"""
    request_id: str = Field(..., description="Request identifier")
    current_step: str = Field(..., description="Current processing step")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated time remaining in seconds")
    current_iteration: Optional[int] = Field(None, description="Current iteration number")
    total_iterations: Optional[int] = Field(None, description="Total planned iterations")
    message: str = Field(..., description="Progress message")

class OptimizationError(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request identifier")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp") 