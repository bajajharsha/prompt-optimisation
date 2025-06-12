"""
Core data models for prompt optimization system
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class OptimizationStatus(str, Enum):
    """Status of optimization process"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelConfiguration(BaseModel):
    """Configuration for the model being optimized against"""
    provider: str = Field(..., description="Model provider (e.g., 'anthropic', 'openai', 'google')")
    model_name: str = Field(..., description="Specific model name (e.g., 'claude-3-sonnet', 'gpt-4')")
    
    def to_string(self) -> str:
        """Get a readable string representation"""
        base = f"{self.provider}/{self.model_name}"
        return base


class EvaluationMetrics(BaseModel):
    """Performance metrics for prompt evaluation"""
    baseline_metrics: Dict[str, Any] = Field(..., description="Baseline metrics")
    failed_cases: List[Dict[str, Any]] = Field(..., description="Failed cases")
    failed_cases_summary: Dict[str, Any] = Field(..., description="Failed cases summary")
    
    # Model context for these metrics
    evaluated_with: Optional[ModelConfiguration] = Field(None, description="Model used for this evaluation")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class PromptHistory(BaseModel):
    """Historical prompt with its performance"""
    prompt: str = Field(..., description="The prompt text")
    metrics: EvaluationMetrics = Field(..., description="Performance metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this prompt was tested")
    iteration: int = Field(..., description="Optimization iteration number")
    optimizer_used: Optional[str] = Field(None, description="Which optimizer generated this prompt")
    
    # Model information for this test
    tested_with: Optional[ModelConfiguration] = Field(None, description="Model configuration used for testing")


class OptimizationContext(BaseModel):
    """Complete context for optimization process"""
    
    # Core inputs
    json_schema: Dict[str, Any] = Field(..., description="Expected JSON schema")
    failed_cases_summary: Dict[str, Any] = Field(..., description="Failed cases summary")
    failed_cases: List[Dict[str, Any]] = Field(..., description="Failed cases")
    baseline_metrics: Dict[str, Any] = Field(..., description="Current performance metrics")
    intent: Dict[str, Any] = Field(..., description="What the prompt is supposed to achieve")
    base_prompt: str = Field(..., description="Current prompt to optimize")
    target_model: ModelConfiguration = Field(..., description="Model the prompt will be optimized for")
    
    prompt_history: List[PromptHistory] = Field(default_factory=list, description="Previous optimization attempts")
    human_feedback: List[str] = Field(default_factory=list, description="Human feedback from previous iterations")
    iteration_number: int = Field(default=1, description="Current optimization iteration")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class OptimizerResult(BaseModel):
    """Result from a single optimizer execution"""
    
    optimizer_name: str = Field(..., description="Name of the optimizer that generated this result")
    candidate_prompt: str = Field(..., description="The optimized prompt")
    reasoning: str = Field(..., description="Why this optimization was made")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this optimization")
    changes_made: List[str] = Field(default_factory=list, description="List of specific changes made")
    execution_time: float = Field(..., description="Time taken to generate this result (seconds)")
    
    # Model-aware optimization info
    optimized_for: Optional[ModelConfiguration] = Field(None, description="Target model this optimization considers")
    
    # Metadata for parallel execution
    timestamp: datetime = Field(default_factory=datetime.now, description="When this result was generated")
    status: OptimizationStatus = Field(default=OptimizationStatus.COMPLETED, description="Execution status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class OptimizerSelection(BaseModel):
    """Selected optimizers with reasoning"""
    
    selected_optimizers: List[str] = Field(..., description="List of optimizer names to execute")
    reasoning: str = Field(..., description="Why these optimizers were selected")
    execution_mode: str = Field(default="parallel", description="How to execute optimizers")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this selection")
    
    # For future expansion
    optimizer_arguments: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Arguments for each optimizer"
    )


class OptimizationRequest(BaseModel):
    """Request for optimization"""
    
    context: OptimizationContext = Field(..., description="Optimization context")
    max_optimizers: int = Field(default=3, description="Maximum number of optimizers to run")
    timeout_seconds: int = Field(default=300, description="Timeout for optimization process")
    include_reasoning: bool = Field(default=True, description="Include detailed reasoning in results")


class OptimizationResponse(BaseModel):
    """Response from optimization process"""
    
    status: OptimizationStatus = Field(..., description="Overall optimization status")
    selected_optimizers: OptimizerSelection = Field(..., description="Which optimizers were selected")
    results: List[OptimizerResult] = Field(default_factory=list, description="Results from each optimizer")
    best_result: Optional[OptimizerResult] = Field(None, description="Best performing result")
    total_execution_time: float = Field(..., description="Total time taken for optimization")
    
    # For debugging and monitoring
    timestamp: datetime = Field(default_factory=datetime.now, description="When optimization completed")
    iteration_number: int = Field(..., description="Optimization iteration")


# Token usage tracking for MongoDB
class TokenUsage(BaseModel):
    """Token usage for Claude API calls"""
    
    timestamp: str = Field(..., description="Timestamp in Asia/Kolkata timezone")
    provider: str = Field(default="claude", description="API provider")
    model: str = Field(..., description="Model used")
    input_tokens: int = Field(..., description="Input tokens consumed")
    output_tokens: int = Field(..., description="Output tokens generated")
    total_tokens: int = Field(..., description="Total tokens used")
    file_name: str = Field(..., description="Source file that made the API call")
    component: str = Field(..., description="Which component made the call")
    operation: str = Field(..., description="What operation was performed") 