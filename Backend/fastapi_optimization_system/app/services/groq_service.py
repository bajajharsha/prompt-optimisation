import httpx
import os
import time
from typing import List, Dict, Any
from datetime import datetime
import asyncio
from pymongo import MongoClient
import pytz


class GroqService:
    """Service for handling Groq API calls"""
    
    def __init__(self):
        self.api_key = os.getenv("groq_api_key")
        if not self.api_key:
            raise ValueError("Groq API key not found. Please set groq_api_key environment variable.")
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # MongoDB setup for logging
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["personal_project_log_usage"]
        self.collection = self.db["llm_usage"]

    async def completions(
        self,
        user_prompt: str,
        system_prompt: str,
        model_name: str,  # Made required - no default
        temperature: float = 1.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Single completion call"""
        if not model_name:
            raise ValueError("model_name is required for Groq service")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": model_name,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                
                # Log usage
                await self._log_usage(response_data, model_name, "completions")
                
                return response_data["choices"][0]["message"]["content"]
                
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

    async def batch_completions(
        self,
        prompts: List[str],
        base_prompt: str,
        model_name: str,  # Made required - no default
        component: str = "unknown",
        operation: str = "batch_completions",
        temperature: float = 1.0,
        max_completion_tokens: int = 1024
    ) -> List[str]:
        """Batch completion calls"""
        if not model_name:
            raise ValueError("model_name is required for Groq service")
            
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
            raise ValueError("model_name is required for Groq service")
            
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
                "provider": "groq",
                "model": model,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/Backend/fastapi_optimization_system/app/services/groq_service.py",
                "component": "groq_service",
                "operation": operation
            }
            
            # Use async insertion
            def sync_insert():
                self.collection.insert_one(log_entry)
            
            # Run sync operation in thread pool
            await asyncio.get_event_loop().run_in_executor(None, sync_insert)
            
        except Exception as e:
            print(f"Token logging failed: {e}")


def get_groq_service() -> GroqService:
    """Get Groq service instance"""
    return GroqService() 