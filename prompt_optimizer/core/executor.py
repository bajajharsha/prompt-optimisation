"""
Executor - Parallel and sequential optimizer execution
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.types import OptimizationContext, OptimizerResult, OptimizationStatus
from ..utils.claude_client import ClaudeClient
from ..optimizers.simple_registry import get_optimizer_by_name


class ExecutionResults:
    """Simple container for execution results"""
    
    def __init__(self):
        self.results: List[OptimizerResult] = []
        self.best_result: Optional[OptimizerResult] = None
        self.execution_status: str = "pending"
        self.total_execution_time: float = 0.0
        self.execution_metadata: Dict[str, Any] = {}


class Executor:
    """
    Executes optimizers in parallel or sequential mode
    Handles timeouts, errors, and result aggregation
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client
        self.max_concurrent_optimizers = 5  # Safety limit
        self.default_timeout = 30  # 30 seconds per optimizer
    
    async def execute_optimizers(
        self,
        optimizer_names: List[str],
        context: OptimizationContext,
        parallel: bool = True,
        timeout_seconds: int = None
    ) -> ExecutionResults:
        """
        Execute multiple optimizers and return aggregated results
        """
        
        if not optimizer_names:
            return self._create_empty_results("No optimizers provided")
        
        timeout_seconds = timeout_seconds or self.default_timeout
        
        print(f"🚀 Executing {len(optimizer_names)} optimizers ({'parallel' if parallel else 'sequential'})")
        print(f"   Optimizers: {', '.join(optimizer_names)}")
        print(f"   Timeout: {timeout_seconds}s per optimizer")
        
        start_time = time.time()
        results = ExecutionResults()
        
        try:
            if parallel:
                optimizer_results = await self._execute_parallel(
                    optimizer_names, context, timeout_seconds
                )
            else:
                optimizer_results = await self._execute_sequential(
                    optimizer_names, context, timeout_seconds
                )
            
            # Process results
            results.results = optimizer_results
            results.best_result = self._select_best_result(optimizer_results)
            results.execution_status = "completed"
            results.total_execution_time = time.time() - start_time
            results.execution_metadata = {
                "execution_mode": "parallel" if parallel else "sequential",
                "requested_optimizers": len(optimizer_names),
                "successful_executions": len([r for r in optimizer_results if r.status == OptimizationStatus.COMPLETED]),
                "failed_executions": len([r for r in optimizer_results if r.status == OptimizationStatus.FAILED]),
                "average_confidence": sum(r.confidence for r in optimizer_results) / len(optimizer_results) if optimizer_results else 0.0
            }
            
            print(f"✅ Execution completed in {results.total_execution_time:.2f}s")
            print(f"   Successful: {results.execution_metadata['successful_executions']}")
            print(f"   Failed: {results.execution_metadata['failed_executions']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Execution failed: {str(e)}")
            results.execution_status = "failed"
            results.total_execution_time = time.time() - start_time
            results.execution_metadata = {"error": str(e)}
            return results
    
    async def _execute_parallel(
        self,
        optimizer_names: List[str],
        context: OptimizationContext,
        timeout_seconds: int
    ) -> List[OptimizerResult]:
        """Execute optimizers in parallel"""
        
        # Limit concurrent executions for safety
        if len(optimizer_names) > self.max_concurrent_optimizers:
            print(f"⚠️ Limiting to {self.max_concurrent_optimizers} concurrent optimizers")
            optimizer_names = optimizer_names[:self.max_concurrent_optimizers]
        
        # Create tasks for each optimizer
        tasks = []
        for name in optimizer_names:
            task = asyncio.create_task(
                self._execute_single_optimizer(name, context, timeout_seconds)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        optimizer_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed result for exception
                optimizer_results.append(OptimizerResult(
                    optimizer_name=optimizer_names[i],
                    candidate_prompt=context.base_prompt,
                    reasoning=f"Execution failed: {str(result)}",
                    confidence=0.0,
                    changes_made=[],
                    execution_time=0.0,
                    optimized_for=context.target_model,
                    timestamp=datetime.now(),
                    status=OptimizationStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                optimizer_results.append(result)
        
        return optimizer_results
    
    async def _execute_sequential(
        self,
        optimizer_names: List[str],
        context: OptimizationContext,
        timeout_seconds: int
    ) -> List[OptimizerResult]:
        """Execute optimizers sequentially"""
        
        results = []
        for name in optimizer_names:
            try:
                result = await self._execute_single_optimizer(name, context, timeout_seconds)
                results.append(result)
                print(f"   ✅ {name}: {result.confidence:.2f} confidence")
            except Exception as e:
                print(f"   ❌ {name}: {str(e)}")
                # Create failed result
                results.append(OptimizerResult(
                    optimizer_name=name,
                    candidate_prompt=context.base_prompt,
                    reasoning=f"Sequential execution failed: {str(e)}",
                    confidence=0.0,
                    changes_made=[],
                    execution_time=0.0,
                    optimized_for=context.target_model,
                    timestamp=datetime.now(),
                    status=OptimizationStatus.FAILED,
                    error_message=str(e)
                ))
        
        return results
    
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
            return OptimizerResult(
                optimizer_name=optimizer_name,
                candidate_prompt=context.base_prompt,
                reasoning=f"Optimizer '{optimizer_name}' not found",
                confidence=0.0,
                changes_made=[],
                execution_time=0.0,
                optimized_for=context.target_model,
                timestamp=datetime.now(),
                status=OptimizationStatus.FAILED,
                error_message=f"Optimizer '{optimizer_name}' not found"
            )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                optimizer.optimize(context),
                timeout=timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            return OptimizerResult(
                optimizer_name=optimizer_name,
                candidate_prompt=context.base_prompt,
                reasoning=f"Optimizer timed out after {timeout_seconds}s",
                confidence=0.0,
                changes_made=[],
                execution_time=timeout_seconds,
                optimized_for=context.target_model,
                timestamp=datetime.now(),
                status=OptimizationStatus.FAILED,
                error_message=f"Timeout after {timeout_seconds} seconds"
            )
        
        except Exception as e:
            return OptimizerResult(
                optimizer_name=optimizer_name,
                candidate_prompt=context.base_prompt,
                reasoning=f"Optimizer execution failed: {str(e)}",
                confidence=0.0,
                changes_made=[],
                execution_time=0.0,
                optimized_for=context.target_model,
                timestamp=datetime.now(),
                status=OptimizationStatus.FAILED,
                error_message=str(e)
            )
    
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
    
    def _create_empty_results(self, reason: str) -> ExecutionResults:
        """Create empty results with error message"""
        results = ExecutionResults()
        results.execution_status = "failed"
        results.execution_metadata = {"error": reason}
        return results
    
    async def close(self):
        """Close any resources"""
        if self.claude_client:
            await self.claude_client.close() 