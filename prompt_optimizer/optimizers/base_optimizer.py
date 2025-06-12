"""
Base optimizer interface for prompt optimization
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import time

from ..models.types import OptimizationContext, OptimizerResult, OptimizationStatus


class BaseOptimizer(ABC):
    """
    Abstract base class for all prompt optimizers
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def optimize(
        self, 
        context: OptimizationContext,
        base_prompt: str
    ) -> OptimizerResult:
        """
        Optimize a prompt based on the given context
        
        Args:
            context: Complete optimization context with metrics, failed cases, etc.
            base_prompt: The current prompt to optimize
            
        Returns:
            OptimizerResult with the optimized prompt and metadata
        """
        pass
    
    def _create_result(
        self,
        candidate_prompt: str,
        reasoning: str,
        confidence: float,
        changes_made: list,
        execution_time: float,
        status: OptimizationStatus = OptimizationStatus.COMPLETED,
        error_message: str = None
    ) -> OptimizerResult:
        """
        Helper method to create a standardized OptimizerResult
        """
        return OptimizerResult(
            optimizer_name=self.name,
            candidate_prompt=candidate_prompt,
            reasoning=reasoning,
            confidence=confidence,
            changes_made=changes_made,
            execution_time=execution_time,
            status=status,
            error_message=error_message
        )
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get basic information about this optimizer
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        } 