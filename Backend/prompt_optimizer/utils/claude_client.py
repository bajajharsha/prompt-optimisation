"""
Claude API client with httpx and MongoDB token usage logging
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import pytz
import httpx
from pymongo import MongoClient

from ..models.types import TokenUsage


class ClaudeAPIError(Exception):
    """Custom exception for Claude API errors"""
    pass


class ClaudeClient:
    """
    Async Claude API client with MongoDB logging
    Optimized for parallel execution
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 14000,
        temperature: float = 0.1,
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # HTTP client for async requests
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=20)  # High concurrency pool
        )
        
        # MongoDB setup (following your pattern)
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017/")
            self.db = self.mongo_client["personal_project_log_usage"]
            self.collection = self.db["llm_usage"]
        except Exception as e:
            print(f"Warning: MongoDB connection failed: {e}")
            self.mongo_client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        if self.mongo_client:
            self.mongo_client.close()
    
    def _log_token_usage(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        component: str = "claude_client",
        operation: str = "completion",
        file_name: str = __file__
    ):
        """Log token usage to MongoDB following your existing pattern"""
        if not self.mongo_client:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                "provider": "claude",
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "file_name": file_name,
                "component": component,
                "operation": operation
            }
            self.collection.insert_one(log_entry)
        except Exception as e:
            print(f"Warning: Failed to log token usage: {e}")
    
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        component: str = "unknown",
        operation: str = "completion",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main completion method - async for parallel execution
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            component: Which component is making this call (for logging)
            operation: What operation is being performed (for logging)
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Dict with 'content' and 'usage' keys
        """
        start_time = time.time()
        
        # Prepare request payload
        payload = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": [{"role": "user", "content": messages}]
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        try:
            # Make async API call
            response = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract content and usage
            content = response_data.get("content", [])
            if content and isinstance(content, list):
                content = content[0].get("text", "")
            
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            # Log token usage
            self._log_token_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                component=component,
                operation=operation,
                file_name=f"/Users/harshabajaj/Desktop/PERSONAL_PROJECT/prompt_optimizer/utils/claude_client.py"
            )
            
            execution_time = time.time() - start_time
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "execution_time": execution_time,
                "model": self.model
            }
            
        except httpx.HTTPStatusError as e:
            error_msg = f"Claude API HTTP error: {e.response.status_code} - {e.response.text}"
            print(f"Error: {error_msg}")
            raise ClaudeAPIError(error_msg)
        
        except httpx.TimeoutException:
            error_msg = f"Claude API timeout after {self.timeout} seconds"
            print(f"Error: {error_msg}")
            raise ClaudeAPIError(error_msg)
        
        except Exception as e:
            error_msg = f"Claude API unexpected error: {str(e)}"
            print(f"Error: {error_msg}")
            raise ClaudeAPIError(error_msg)
    
    async def optimize_prompt(
        self, 
        system_prompt: str, 
        user_message: str,
        component: str = "optimizer"
    ) -> Dict[str, Any]:
        """
        Convenience method for prompt optimization calls
        """
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
        ]
        
        return await self.complete(
            messages=messages,
            component=component,
            operation="prompt_optimization"
        )
    
    async def analyze_context(
        self, 
        system_prompt: str, 
        context_data: str,
        component: str = "orchestrator"
    ) -> Dict[str, Any]:
        """
        Convenience method for context analysis calls
        """
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nContext to analyze:\n{context_data}"}
        ]
        
        return await self.complete(
            messages=messages,
            component=component,
            operation="context_analysis"
        )
    
    # Synchronous wrapper methods for backwards compatibility
    def complete_sync(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Synchronous version of complete method"""
        return asyncio.run(self.complete(messages, **kwargs))
    
    def optimize_prompt_sync(self, system_prompt: str, user_message: str, **kwargs) -> Dict[str, Any]:
        """Synchronous version of optimize_prompt method"""
        return asyncio.run(self.optimize_prompt(system_prompt, user_message, **kwargs))


# Utility function for parallel Claude calls
async def run_parallel_completions(
    client: ClaudeClient,
    completion_requests: List[Dict[str, Any]],
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Run multiple Claude API calls in parallel with concurrency control
    
    Args:
        client: ClaudeClient instance
        completion_requests: List of dicts with 'messages', 'component', 'operation' keys
        max_concurrent: Maximum concurrent requests
    
    Returns:
        List of completion results in same order as requests
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_complete(request: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await client.complete(**request)
    
    # Create tasks for all requests
    tasks = [bounded_complete(request) for request in completion_requests]
    
    # Execute in parallel and return results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error dicts
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({
                "error": str(result),
                "content": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "execution_time": 0
            })
        else:
            processed_results.append(result)
    
    return processed_results 