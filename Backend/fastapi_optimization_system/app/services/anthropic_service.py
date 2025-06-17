import httpx
import os
import time
from typing import List, Dict, Any
from datetime import datetime
import asyncio
from pymongo import MongoClient
import pytz


class AnthropicService:
    """Service for handling Anthropic Claude API calls"""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("anthropic_api_key")
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY or anthropic_api_key environment variable.")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # MongoDB setup for logging
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["personal_project_log_usage"]
        self.collection = self.db["llm_usage"]

    async def completions(
        self,
        user_prompt: str,
        system_prompt: str,
        model_name: str,  # Made required - no default
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Single completion call"""
        if not model_name:
            raise ValueError("model_name is required for Anthropic service")
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                
                # Log usage
                await self._log_usage(response_data, model_name, "completions")
                
                return response_data["content"][0]["text"]
                
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

    async def batch_completions(
        self,
        prompts: List[str],
        base_prompt: str,
        model_name: str,  # Made required - no default
        component: str = "unknown",
        operation: str = "batch_completions",
        temperature: float = 0.2,
        max_completion_tokens: int = 1024
    ) -> List[str]:
        """Batch completion calls"""
        if not model_name:
            raise ValueError("model_name is required for Anthropic service")
            
        responses = []
        
        for prompt in prompts:
            try:
                response = await self.completions(
                    user_prompt=prompt,
                    system_prompt=base_prompt,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_completion_tokens
                )
                responses.append(response)
            except Exception as e:
                print(f"Error in batch completion: {e}")
                responses.append("")  # Add empty string on error
        
        return responses

    async def inference_with_system_user_prompts(
        self,
        system_prompt: str,
        user_prompts: List[str],
        model_name: str,  # Made required - no default
        component: str = "unknown",
        operation: str = "system_user_inference",
        temperature: float = 0.2,
        max_completion_tokens: int = 1024
    ) -> List[str]:
        """Inference with separate system and user prompts"""
        if not model_name:
            raise ValueError("model_name is required for Anthropic service")
            
        responses = []
        
        for user_prompt in user_prompts:
            try:
                response = await self.completions(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_completion_tokens
                )
                responses.append(response)
            except Exception as e:
                print(f"Error in system/user inference: {e}")
                responses.append("")  # Add empty string on error
        
        return responses

    async def _log_usage(self, response_data: Dict[str, Any], model: str, operation: str):
        """Log token usage to MongoDB"""
        try:
            usage = response_data.get("usage", {})
            log_entry = {
                "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                "provider": "anthropic",
                "model": model,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/Backend/fastapi_optimization_system/app/services/anthropic_service.py",
                "component": "anthropic_service",
                "operation": operation
            }
            
            # Use async insertion
            def sync_insert():
                self.collection.insert_one(log_entry)
            
            # Run sync operation in thread pool
            await asyncio.get_event_loop().run_in_executor(None, sync_insert)
            
        except Exception as e:
            print(f"Token logging failed: {e}")


def get_anthropic_service() -> AnthropicService:
    """Get Anthropic service instance"""
    return AnthropicService() 