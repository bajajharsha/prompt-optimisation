import json
import time 
import base64
import os
import httpx
from typing import List, Dict, Any
from datetime import datetime
import asyncio
from pymongo import MongoClient
import pytz


class GeminiAPIService:
    """Service for handling Google Gemini API calls using direct REST API"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY or gemini_api_key environment variable.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # MongoDB setup for logging
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["personal_project_log_usage"]
        self.collection = self.db["llm_usage"]

    async def completions(
        self,
        user_prompt: str,
        system_prompt: str,
        model_name: str,  # Made required - no default
        temperature: float = 0.1,
        max_tokens: int = 40000,
        **kwargs
    ) -> str:
        """Single completion call for text-only content using REST API"""
        if not model_name:
            raise ValueError("model_name is required for Gemini service")
        
        try:
            # Prepare the API request
            url = f"{self.base_url}/models/{model_name}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            # Combine system and user prompt for Gemini
            combined_prompt = f"{system_prompt}\n\nUser input: {user_prompt}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": combined_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "candidateCount": 1
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
            
            # Extract response text
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    response_text = candidate["content"]["parts"][0]["text"]
                else:
                    response_text = ""
            else:
                response_text = ""
            
            # Log usage
            await self._log_usage(response_data, model_name, "completions")
            
            return response_text
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    async def completions_with_system_instruction(
        self,
        user_prompt: str,
        system_prompt: str,
        model_name: str,  # Made required - no default
        temperature: float = 0.1,
        max_tokens: int = 40000,
        **kwargs
    ) -> str:
        """Single completion call using system instruction feature"""
        if not model_name:
            raise ValueError("model_name is required for Gemini service")
        
        try:
            # Prepare the API request
            url = f"{self.base_url}/models/{model_name}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            payload = {
                "systemInstruction": {
                    "parts": [
                        {
                            "text": system_prompt
                        }
                    ]
                },
                "contents": [
                    {
                        "parts": [
                            {
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "candidateCount": 1
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
            
            # Extract response text
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    response_text = candidate["content"]["parts"][0]["text"]
                else:
                    response_text = ""
            else:
                response_text = ""
            
            # Log usage
            await self._log_usage(response_data, model_name, "completions_with_system_instruction")
            
            return response_text
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    async def batch_completions(
        self,
        prompts: List[str],
        base_prompt: str,
        model_name: str,  # Made required - no default
        component: str = "unknown",
        operation: str = "batch_completions",
        temperature: float = 0.1,
        max_completion_tokens: int = 40000
    ) -> List[str]:
        """Batch completion calls"""
        if not model_name:
            raise ValueError("model_name is required for Gemini service")
            
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
        temperature: float = 0.1,
        max_completion_tokens: int = 40000
    ) -> List[str]:
        """Inference with separate system and user prompts using system instruction"""
        if not model_name:
            raise ValueError("model_name is required for Gemini service")
            
        responses = []
        
        for user_prompt in user_prompts:
            try:
                response = await self.completions_with_system_instruction(
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

    async def generate_content_with_image(
        self, prompt: str, image_path: str, temperature: float = 0.1
    ) -> str:
        """Generate content with image input using REST API"""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format
            image_format = "image/jpeg"
            if image_path.lower().endswith('.png'):
                image_format = "image/png"
            elif image_path.lower().endswith('.webp'):
                image_format = "image/webp"
            
            # Prepare the API request
            url = f"{self.base_url}/models/{self.default_model}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": image_format,
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 40000,
                    "candidateCount": 1
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
            
            # Extract response text
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    response_text = candidate["content"]["parts"][0]["text"]
                else:
                    response_text = ""
            else:
                response_text = ""
            
            # Log usage
            await self._log_usage(response_data, self.default_model, "generate_content_with_image")
            
            return response_text
            
        except FileNotFoundError:
            raise Exception(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
            
    async def generate_stream_content_with_image(
        self, prompt: str, image_path: str, temperature: float = 0.1
    ):
        """Generate streaming content with image input using REST API"""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format
            image_format = "image/jpeg"
            if image_path.lower().endswith('.png'):
                image_format = "image/png"
            elif image_path.lower().endswith('.webp'):
                image_format = "image/webp"
            
            # Prepare the API request for streaming
            url = f"{self.base_url}/models/{self.default_model}:streamGenerateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": image_format,
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 40000,
                    "candidateCount": 1
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                # Parse each streaming response
                                chunk_data = json.loads(line)
                                if "candidates" in chunk_data and len(chunk_data["candidates"]) > 0:
                                    candidate = chunk_data["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        chunk_text = candidate["content"]["parts"][0]["text"]
                                        yield chunk_text
                            except json.JSONDecodeError:
                                continue
                        
        except FileNotFoundError:
            raise Exception(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    async def _log_usage(self, response_data: Dict[str, Any], model: str, operation: str):
        """Log token usage to MongoDB"""
        try:
            # Extract usage information from response
            usage_metadata = response_data.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount", 0)
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)
            total_tokens = usage_metadata.get("totalTokenCount", input_tokens + output_tokens)
            
            log_entry = {
                "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                "provider": "google",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/Backend/fastapi_optimization_system/app/services/gemini_service.py",
                "component": "gemini_service",
                "operation": operation
            }
            
            # Use async insertion
            def sync_insert():
                self.collection.insert_one(log_entry)
            
            # Run sync operation in thread pool
            await asyncio.get_event_loop().run_in_executor(None, sync_insert)
            
        except Exception as e:
            print(f"Token logging failed: {e}")


def get_gemini_service() -> GeminiAPIService:
    """Get Gemini service instance"""
    return GeminiAPIService()