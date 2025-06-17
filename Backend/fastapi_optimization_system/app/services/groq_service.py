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
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        try:
            # Configure SSL context to handle certificate issues
            async with httpx.AsyncClient(
                timeout=60.0,
                verify=False  # Disable SSL verification to handle certificate issues
            ) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                
                # Log usage
                await self._log_usage(response_data, model_name, "completions")
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    return response_data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Invalid response format: {response_data}")
                
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
            raise Exception(f"Groq API HTTP error {e.response.status_code}: {error_detail}")
        except httpx.RequestError as e:
            raise Exception(f"Groq API request error: {str(e)}")
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
        max_completion_tokens: int = 1024,
        max_concurrent: int = 10  # Optimized for concurrent API calls
    ) -> List[str]:
        """Batch completion calls with parallel processing for speed"""
        if not model_name:
            raise ValueError("model_name is required for Groq service")
        
        if not prompts:
            return []
            
        print(f"🚀 Processing {len(prompts)} prompts in parallel batches (max {max_concurrent} concurrent)")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_prompt(prompt_idx: int, prompt: str) -> str:
            async with semaphore:
                try:
                    response = await self.completions(
                        user_prompt=prompt,
                        system_prompt=base_prompt,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_completion_tokens
                    )
                    return response
                except Exception as e:
                    print(f"Error in batch completion for prompt {prompt_idx}: {e}")
                    return ""  # Return empty string on error
        
        # Create tasks for all prompts
        tasks = [process_single_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Execute all tasks in parallel
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # Handle exceptions and convert to strings
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"Exception in prompt {i}: {response}")
                final_responses.append("")
            else:
                final_responses.append(response)
        
        successful_count = len([r for r in final_responses if r])
        print(f"✅ Batch completed in {processing_time:.1f}s: {successful_count}/{len(prompts)} successful")
        
        return final_responses

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