"""
Simple optimizer registry for managing available optimizers
"""

from typing import Dict, List, Optional, Any
from .base_optimizer import BaseOptimizer
from .freeform_optimizer import FreeformOptimizer
from ..utils.claude_client import ClaudeClient


class OptimizerRegistry:
    """
    Simple registry to manage and provide access to optimizers
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client
        self._optimizers: Dict[str, BaseOptimizer] = {}
        
        # Register default optimizers
        self._register_default_optimizers()
    
    def _register_default_optimizers(self):
        """Register the default optimizers"""
        # For now, only freeform optimizer
        # Note: Claude client will be created when needed
        freeform = FreeformOptimizer(claude_client=self.claude_client)
        self.register_optimizer(freeform)
    
    def register_optimizer(self, optimizer: BaseOptimizer) -> None:
        """
        Register a new optimizer
        
        Args:
            optimizer: The optimizer instance to register
        """
        self._optimizers[optimizer.name] = optimizer
        print(f"Registered optimizer: {optimizer.name}")
    
    def get_optimizer(self, name: str) -> Optional[BaseOptimizer]:
        """
        Get an optimizer by name
        
        Args:
            name: Name of the optimizer
            
        Returns:
            Optimizer instance or None if not found
        """
        return self._optimizers.get(name)
    
    def get_all_optimizers(self) -> Dict[str, BaseOptimizer]:
        """
        Get all registered optimizers
        
        Returns:
            Dictionary of optimizer name to optimizer instance
        """
        return self._optimizers.copy()
    
    def list_optimizer_names(self) -> List[str]:
        """
        Get list of all registered optimizer names
        
        Returns:
            List of optimizer names
        """
        return list(self._optimizers.keys())
    
    def get_optimizer_info(self, name: str = None) -> Dict[str, Any]:
        """
        Get information about optimizers
        
        Args:
            name: Specific optimizer name, or None for all optimizers
            
        Returns:
            Dictionary with optimizer information
        """
        if name:
            optimizer = self.get_optimizer(name)
            if optimizer:
                return optimizer.get_info()
            else:
                return {"error": f"Optimizer '{name}' not found"}
        else:
            # Return info for all optimizers
            return {
                name: optimizer.get_info() 
                for name, optimizer in self._optimizers.items()
            }
    
    def is_registered(self, name: str) -> bool:
        """
        Check if an optimizer is registered
        
        Args:
            name: Optimizer name to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._optimizers
    
    def unregister_optimizer(self, name: str) -> bool:
        """
        Unregister an optimizer
        
        Args:
            name: Name of optimizer to unregister
            
        Returns:
            True if successfully unregistered, False if not found
        """
        if name in self._optimizers:
            del self._optimizers[name]
            print(f"Unregistered optimizer: {name}")
            return True
        return False
    
    def get_default_optimizer(self) -> BaseOptimizer:
        """
        Get the default optimizer (freeform)
        
        Returns:
            Default optimizer instance
        """
        return self.get_optimizer("freeform")
    
    def validate_optimizer_selection(self, optimizer_names: List[str]) -> Dict[str, Any]:
        """
        Validate a list of optimizer names
        
        Args:
            optimizer_names: List of optimizer names to validate
            
        Returns:
            Dictionary with validation results
        """
        valid_optimizers = []
        invalid_optimizers = []
        
        for name in optimizer_names:
            if self.is_registered(name):
                valid_optimizers.append(name)
            else:
                invalid_optimizers.append(name)
        
        return {
            "valid": valid_optimizers,
            "invalid": invalid_optimizers,
            "all_valid": len(invalid_optimizers) == 0
        }
    
    async def close(self):
        """Close any resources used by optimizers"""
        if self.claude_client:
            await self.claude_client.close()


# Global registry instance for easy access
_global_registry: Optional[OptimizerRegistry] = None


def get_global_registry() -> OptimizerRegistry:
    """
    Get the global optimizer registry instance
    
    Returns:
        Global OptimizerRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = OptimizerRegistry()
    return _global_registry


def register_optimizer_globally(optimizer: BaseOptimizer) -> None:
    """
    Register an optimizer in the global registry
    
    Args:
        optimizer: Optimizer to register
    """
    registry = get_global_registry()
    registry.register_optimizer(optimizer)


def get_optimizer_by_name(name: str) -> Optional[BaseOptimizer]:
    """
    Get an optimizer by name from global registry
    
    Args:
        name: Optimizer name
        
    Returns:
        Optimizer instance or None
    """
    registry = get_global_registry()
    return registry.get_optimizer(name) 