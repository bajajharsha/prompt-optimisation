"""
Optimizer Executor - Manages parallel execution of optimization strategies
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import traceback

from ..models.types import (
    OptimizationContext, 
    OptimizerSelection,
    OptimizerResult,
    OptimizationStatus,
    OptimizationResponse
)
from ..optimizers.registry import get_global_registry
from ..utils.claude_client import ClaudeClient


class OptimizerExecutor:
    """
    Manages parallel execution of optimization strategies
    Handles timeouts, error recovery, and result comparison
    """
    
    def __init__(self, claude_client: ClaudeClient = None):
        self.claude_client = claude_client
        self.optimizer_registry = get_global_registry()
        
        # Execution parameters
        self.default_timeout = 300  # 5 minutes default timeout
        self.max_parallel_executions = 5  # Safety limit
        self.result_comparison_enabled = True
        
    async def execute_optimization_strategies(
        self,
        context: OptimizationContext,
        selection: OptimizerSelection,
        timeout_seconds: Optional[int] = None
    ) -> OptimizationResponse:
        """
        Execute the selected optimization strategies in parallel
        
        Args:
            context: Optimization context
            selection: Selected optimizers and execution plan
            timeout_seconds: Timeout for entire execution (defaults to self.default_timeout)
            
        Returns:
            OptimizationResponse with all results and best selection
        """
        
        start_time = time.time()
        timeout = timeout_seconds or self.default_timeout
        
        print(f"🚀 Executing {len(selection.selected_optimizers)} optimizers in parallel...")
        print(f"   Optimizers: {', '.join(selection.selected_optimizers)}")
        print(f"   Timeout: {timeout} seconds")
        print(f"   Execution mode: {selection.execution_mode}")
        
        try:
            # Execute optimizers based on execution mode
            if selection.execution_mode == "parallel":
                results = await self._execute_parallel(
                    context, selection.selected_optimizers, timeout
                )
            elif selection.execution_mode == "sequential":
                results = await self._execute_sequential(
                    context, selection.selected_optimizers, timeout
                )
            else:
                # Default to parallel
                results = await self._execute_parallel(
                    context, selection.selected_optimizers, timeout
                )
            
            # Find best result
            best_result = self._select_best_result(results, context)
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Determine overall status
            overall_status = self._determine_overall_status(results)
            
            # Create response
            response = OptimizationResponse(
                status=overall_status,
                selected_optimizers=selection,
                results=results,
                best_result=best_result,
                total_execution_time=total_time,
                timestamp=datetime.now(),
                iteration_number=context.iteration_number
            )
            
            print(f"✅ Execution completed in {total_time:.2f}s")
            print(f"   Status: {overall_status}")
            print(f"   Results: {len(results)} optimizers executed")
            print(f"   Best result: {best_result.optimizer_name if best_result else 'None'}")
            
            return response
            
        except asyncio.TimeoutError:
            total_time = time.time() - start_time
            print(f"⏰ Execution timed out after {total_time:.2f}s")
            
            return OptimizationResponse(
                status=OptimizationStatus.FAILED,
                selected_optimizers=selection,
                results=[],
                best_result=None,
                total_execution_time=total_time,
                timestamp=datetime.now(),
                iteration_number=context.iteration_number
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Execution failed: {str(e)}")
            
            return OptimizationResponse(
                status=OptimizationStatus.FAILED,
                selected_optimizers=selection,
                results=[],
                best_result=None,
                total_execution_time=total_time,
                timestamp=datetime.now(),
                iteration_number=context.iteration_number
            )
    
    async def _execute_parallel(
        self,
        context: OptimizationContext,
        optimizer_names: List[str],
        timeout: int
    ) -> List[OptimizerResult]:
        """Execute optimizers in parallel"""
        
        print(f"⚡ Starting parallel execution of {len(optimizer_names)} optimizers...")
        
        # Create tasks for each optimizer
        tasks = []
        for optimizer_name in optimizer_names:
            task = self._execute_single_optimizer(context, optimizer_name)
            tasks.append(task)
        
        # Execute all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create failed result for exception
                    failed_result = self._create_failed_result(
                        optimizer_names[i], 
                        str(result),
                        context.base_prompt
                    )
                    processed_results.append(failed_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            print(f"⏰ Parallel execution timed out after {timeout}s")
            raise
    
    async def _execute_sequential(
        self,
        context: OptimizationContext,
        optimizer_names: List[str],
        timeout: int
    ) -> List[OptimizerResult]:
        """Execute optimizers sequentially"""
        
        print(f"🔄 Starting sequential execution of {len(optimizer_names)} optimizers...")
        
        results = []
        start_time = time.time()
        
        for i, optimizer_name in enumerate(optimizer_names):
            # Check if we're running out of time
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                print(f"⏰ Sequential execution timed out after {i} optimizers")
                break
            
            try:
                print(f"   Executing {optimizer_name} ({i+1}/{len(optimizer_names)})...")
                result = await asyncio.wait_for(
                    self._execute_single_optimizer(context, optimizer_name),
                    timeout=remaining
                )
                results.append(result)
                
                print(f"   ✅ {optimizer_name} completed in {result.execution_time:.2f}s")
                
            except asyncio.TimeoutError:
                print(f"   ⏰ {optimizer_name} timed out")
                failed_result = self._create_failed_result(
                    optimizer_name,
                    f"Execution timed out after {remaining:.1f}s",
                    context.base_prompt
                )
                results.append(failed_result)
                
            except Exception as e:
                print(f"   ❌ {optimizer_name} failed: {str(e)}")
                failed_result = self._create_failed_result(
                    optimizer_name,
                    str(e),
                    context.base_prompt
                )
                results.append(failed_result)
        
        return results
    
    async def _execute_single_optimizer(
        self,
        context: OptimizationContext,
        optimizer_name: str
    ) -> OptimizerResult:
        """Execute a single optimizer"""
        
        start_time = time.time()
        
        try:
            # Get optimizer from registry
            optimizer = self.optimizer_registry.get_optimizer(optimizer_name)
            if optimizer is None:
                raise ValueError(f"Optimizer '{optimizer_name}' not found in registry")
            
            # Ensure optimizer has Claude client if needed
            if hasattr(optimizer, 'claude_client') and optimizer.claude_client is None:
                if self.claude_client:
                    optimizer.claude_client = self.claude_client
                else:
                    # Create a new one (will fail if no API key, which is expected)
                    optimizer.claude_client = ClaudeClient()
            
            # Execute optimization
            result = await optimizer.optimize(context, context.base_prompt)
            
            # Validate result
            if not isinstance(result, OptimizerResult):
                raise ValueError(f"Optimizer {optimizer_name} returned invalid result type")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ⚠️ {optimizer_name} failed: {str(e)}")
            
            return self._create_failed_result(
                optimizer_name,
                str(e),
                context.base_prompt,
                execution_time
            )
    
    def _create_failed_result(
        self,
        optimizer_name: str,
        error_message: str,
        base_prompt: str,
        execution_time: float = 0.0
    ) -> OptimizerResult:
        """Create a failed result for error cases"""
        
        return OptimizerResult(
            optimizer_name=optimizer_name,
            candidate_prompt=base_prompt,  # Return original prompt
            reasoning=f"Optimization failed: {error_message}",
            confidence=0.0,
            changes_made=[],
            execution_time=execution_time,
            timestamp=datetime.now(),
            status=OptimizationStatus.FAILED,
            error_message=error_message
        )
    
    def _select_best_result(
        self,
        results: List[OptimizerResult],
        context: OptimizationContext
    ) -> Optional[OptimizerResult]:
        """Select the best result from all optimizer results"""
        
        if not results:
            return None
        
        # Filter successful results
        successful_results = [r for r in results if r.status == OptimizationStatus.COMPLETED]
        
        if not successful_results:
            print("   ⚠️ No successful optimization results")
            return None
        
        print(f"   🔍 Comparing {len(successful_results)} successful results...")
        
        # Simple selection strategy: highest confidence
        # In the future, this could be more sophisticated with actual evaluation
        best_result = max(successful_results, key=lambda r: r.confidence)
        
        print(f"   🏆 Best result: {best_result.optimizer_name} (confidence: {best_result.confidence:.3f})")
        
        return best_result
    
    def _determine_overall_status(self, results: List[OptimizerResult]) -> OptimizationStatus:
        """Determine overall optimization status from individual results"""
        
        if not results:
            return OptimizationStatus.FAILED
        
        successful_count = sum(1 for r in results if r.status == OptimizationStatus.COMPLETED)
        failed_count = sum(1 for r in results if r.status == OptimizationStatus.FAILED)
        
        if successful_count > 0:
            return OptimizationStatus.COMPLETED
        elif failed_count == len(results):
            return OptimizationStatus.FAILED
        else:
            return OptimizationStatus.PENDING  # Shouldn't happen in normal flow
    
    async def execute_single_optimizer_test(
        self,
        context: OptimizationContext,
        optimizer_name: str,
        timeout_seconds: int = 60
    ) -> OptimizerResult:
        """
        Execute a single optimizer for testing purposes
        
        Args:
            context: Optimization context
            optimizer_name: Name of optimizer to test
            timeout_seconds: Timeout for execution
            
        Returns:
            OptimizerResult from the execution
        """
        
        print(f"🧪 Testing single optimizer: {optimizer_name}")
        print(f"   Timeout: {timeout_seconds} seconds")
        
        try:
            result = await asyncio.wait_for(
                self._execute_single_optimizer(context, optimizer_name),
                timeout=timeout_seconds
            )
            
            print(f"✅ Test completed in {result.execution_time:.2f}s")
            print(f"   Status: {result.status}")
            print(f"   Confidence: {result.confidence:.3f}")
            
            return result
            
        except asyncio.TimeoutError:
            print(f"⏰ Test timed out after {timeout_seconds}s")
            return self._create_failed_result(
                optimizer_name,
                f"Test execution timed out after {timeout_seconds}s",
                context.base_prompt
            )
        
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return self._create_failed_result(
                optimizer_name,
                str(e),
                context.base_prompt
            )
    
    def get_execution_stats(self, response: OptimizationResponse) -> Dict[str, Any]:
        """Get detailed execution statistics from response"""
        
        results = response.results
        
        stats = {
            "total_optimizers": len(response.selected_optimizers.selected_optimizers),
            "executed_optimizers": len(results),
            "successful_optimizers": sum(1 for r in results if r.status == OptimizationStatus.COMPLETED),
            "failed_optimizers": sum(1 for r in results if r.status == OptimizationStatus.FAILED),
            "total_execution_time": response.total_execution_time,
            "average_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0.0,
            "fastest_optimizer": min(results, key=lambda r: r.execution_time).optimizer_name if results else None,
            "slowest_optimizer": max(results, key=lambda r: r.execution_time).optimizer_name if results else None,
            "highest_confidence": max(results, key=lambda r: r.confidence).optimizer_name if results else None,
            "confidence_scores": {r.optimizer_name: r.confidence for r in results},
            "execution_times": {r.optimizer_name: r.execution_time for r in results},
            "status_breakdown": {
                "completed": [r.optimizer_name for r in results if r.status == OptimizationStatus.COMPLETED],
                "failed": [r.optimizer_name for r in results if r.status == OptimizationStatus.FAILED]
            }
        }
        
        return stats
    
    async def validate_optimizers(self, optimizer_names: List[str]) -> Dict[str, Any]:
        """Validate that optimizers are available and ready"""
        
        validation_result = {
            "valid_optimizers": [],
            "invalid_optimizers": [],
            "optimizer_info": {},
            "all_valid": True
        }
        
        for name in optimizer_names:
            optimizer = self.optimizer_registry.get_optimizer(name)
            if optimizer is None:
                validation_result["invalid_optimizers"].append(name)
                validation_result["all_valid"] = False
            else:
                validation_result["valid_optimizers"].append(name)
                validation_result["optimizer_info"][name] = {
                    "name": optimizer.name,
                    "description": optimizer.description,
                    "has_claude_client": hasattr(optimizer, 'claude_client') and optimizer.claude_client is not None
                }
        
        return validation_result
    
    async def close(self):
        """Close any resources"""
        if self.claude_client:
            await self.claude_client.close() 