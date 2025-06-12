"""
Simple Executor - Concurrent optimizer execution
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.types import OptimizationContext, OptimizerResult, OptimizationStatus
from ..utils.claude_client import ClaudeClient
from ..optimizers.simple_registry import get_optimizer_by_name


class ExecutionResults:
    """Container for execution results"""
    
    def __init__(self):
        self.results: List[OptimizerResult] = []
        self.best_result: Optional[OptimizerResult] = None
        self.execution_time: float = 0.0
        self.successful_count: int = 0
        self.failed_count: int = 0


class SimpleExecutor:
    """
    Executes multiple optimizers concurrently (all at same time)
    Perfect for API-based optimizers that benefit from async execution
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client
    
    async def execute_optimizers(
        self,
        optimizer_names: List[str],
        context: OptimizationContext,
    ) -> ExecutionResults:
        """
        Execute multiple optimizers concurrently and return best result
        
        Args:
            optimizer_names: List of optimizer names to run
            context: Optimization context
            timeout_seconds: Timeout per optimizer (default: 30s)
            
        Returns:
            ExecutionResults with all results and best selection
        """
        
        if not optimizer_names:
            return self._create_empty_results("No optimizers provided")
        
        
        print(f"🚀 Running {len(optimizer_names)} optimizers concurrently")
        print(f"   Optimizers: {', '.join(optimizer_names)}")
        
        start_time = time.time()
        
        try:
            # Create tasks for all optimizers
            tasks = []
            for name in optimizer_names:
                task = asyncio.create_task(
                    self._execute_single_optimizer(name, context),
                    name=f"optimizer_{name}"
                )
                tasks.append(task)
            
            # Run all optimizers concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            execution_results = self._process_results(results, optimizer_names)
            execution_results.execution_time = time.time() - start_time
            
            print(f"✅ Execution completed in {execution_results.execution_time:.2f}s")
            print(f"   Successful: {execution_results.successful_count}")
            print(f"   Failed: {execution_results.failed_count}")
            
            if execution_results.best_result:
                print(f"   Best: {execution_results.best_result.optimizer_name} (confidence: {execution_results.best_result.confidence:.2f})")
            
            return execution_results
            
        except Exception as e:
            print(f"❌ Execution failed: {str(e)}")
            return self._create_empty_results(f"Execution error: {str(e)}")
    
    async def _execute_single_optimizer(
        self,
        optimizer_name: str,
        context: OptimizationContext,
        timeout_seconds: int
    ) -> OptimizerResult:
        """Execute a single optimizer with timeout"""
        
        # Get optimizer instance
        optimizer = get_optimizer_by_name(optimizer_name, self.claude_client)
        
        if not optimizer:
            return self._create_failed_result(
                optimizer_name, 
                f"Optimizer '{optimizer_name}' not found",
                context
            )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                optimizer.optimize(context),
                timeout=timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            return self._create_failed_result(
                optimizer_name,
                f"Optimizer timed out after {timeout_seconds}s",
                context,
                timeout_seconds
            )
        
        except Exception as e:
            return self._create_failed_result(
                optimizer_name,
                f"Optimizer execution failed: {str(e)}",
                context
            )
    
    def _process_results(
        self, 
        raw_results: List, 
        optimizer_names: List[str]
    ) -> ExecutionResults:
        """Process raw results and exceptions into ExecutionResults"""
        
        execution_results = ExecutionResults()
        
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                # Handle exceptions from asyncio.gather
                failed_result = self._create_failed_result(
                    optimizer_names[i],
                    f"Execution exception: {str(result)}",
                    None  # We don't have context here
                )
                execution_results.results.append(failed_result)
                execution_results.failed_count += 1
            else:
                execution_results.results.append(result)
                if result.status == OptimizationStatus.COMPLETED:
                    execution_results.successful_count += 1
                else:
                    execution_results.failed_count += 1
        
        # Select best result
        execution_results.best_result = self._select_best_result(execution_results.results)
        
        return execution_results
    
    def _select_best_result(self, results: List[OptimizerResult]) -> Optional[OptimizerResult]:
        """Select the best result based on confidence scores"""
        
        if not results:
            return None
        
        # Filter successful results
        successful_results = [r for r in results if r.status == OptimizationStatus.COMPLETED]
        
        if not successful_results:
            return None
        
        # Return result with highest confidence
        return max(successful_results, key=lambda r: r.confidence)
    
    def _create_failed_result(
        self,
        optimizer_name: str,
        error_message: str,
        context: OptimizationContext,
        execution_time: float = 0.0
    ) -> OptimizerResult:
        """Create a failed result for error cases"""
        
        return OptimizerResult(
            optimizer_name=optimizer_name,
            candidate_prompt=context.base_prompt if context else "",
            reasoning=f"Execution failed: {error_message}",
            confidence=0.0,
            changes_made=[],
            execution_time=execution_time,
            optimized_for=context.target_model if context else None,
            timestamp=datetime.now(),
            status=OptimizationStatus.FAILED,
            error_message=error_message
        )
    
    def _create_empty_results(self, reason: str) -> ExecutionResults:
        """Create empty results with error message"""
        results = ExecutionResults()
        results.failed_count = 1
        print(f"❌ {reason}")
        return results
    
    async def close(self):
        """Close any resources"""
        if self.claude_client:
            await self.claude_client.close()


if __name__ == "__main__":
    print("🔧 Simple Executor - Concurrent Optimization")
    print("=" * 50)
    print("This executor runs multiple optimizers concurrently")
    print("Perfect for API-based optimizers that benefit from async execution")
    print("No need for sequential mode - concurrent is always better for API calls")
    print("✅ Simple and focused!") 