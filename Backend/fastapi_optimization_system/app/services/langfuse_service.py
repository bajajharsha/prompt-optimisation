import httpx
import json
import csv
import io
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from fastapi import UploadFile

from app.config.settings import get_settings
from app.utils.error_handler import DatasetError
from app.utils.context_util import get_request_id


class LangFuseService:
    """
    Service for interacting with LangFuse API
    Handles low-level API calls and payload preparation
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.langfuse_host
        self.public_key = self.settings.langfuse_public_key
        self.secret_key = self.settings.langfuse_secret_key
        
        if not self.public_key or not self.secret_key:
            raise ValueError("LangFuse API keys are required. Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    
    async def create_dataset(self, dataset_name: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new dataset in LangFuse using the v2 API
        
        Args:
            dataset_name: Name of the dataset
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            Dict containing the created dataset information
            
        Raises:
            DatasetError: If dataset creation fails
        """
        request_id = get_request_id()
        
        try:
            # Prepare payload for LangFuse v2 API
            payload = {
                "name": dataset_name
            }
            
            if description:
                payload["description"] = description
                
            if metadata:
                payload["metadata"] = metadata
            
            # Make API call to create dataset
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/public/v2/datasets",
                    json=payload,
                    auth=(self.public_key, self.secret_key),
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    print(f"✅ Dataset '{dataset_name}' created successfully in LangFuse")
                    return result
                elif response.status_code == 409:
                    # Dataset already exists - this might be acceptable
                    print(f"⚠️  Dataset '{dataset_name}' already exists in LangFuse")
                    # Try to get the existing dataset
                    return await self.get_dataset(dataset_name)
                else:
                    error_detail = response.text
                    raise DatasetError(
                        f"Failed to create dataset in LangFuse: {response.status_code} - {error_detail}",
                        request_id=request_id,
                        dataset_name=dataset_name
                    )
                    
        except httpx.HTTPError as e:
            raise DatasetError(
                f"HTTP error while creating dataset: {str(e)}",
                request_id=request_id,
                dataset_name=dataset_name
            )
        except Exception as e:
            raise DatasetError(
                f"Unexpected error while creating dataset: {str(e)}",
                request_id=request_id,
                dataset_name=dataset_name
            )
    
    async def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get an existing dataset from LangFuse
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing the dataset information
        """
        request_id = get_request_id()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/public/v2/datasets/{dataset_name}",
                    auth=(self.public_key, self.secret_key),
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise DatasetError(
                        f"Dataset '{dataset_name}' not found in LangFuse",
                        request_id=request_id,
                        dataset_name=dataset_name
                    )
                    
        except httpx.HTTPError as e:
            raise DatasetError(
                f"HTTP error while fetching dataset: {str(e)}",
                request_id=request_id,
                dataset_name=dataset_name
            )
    
    async def create_dataset_item(self, dataset_name: str, input_data: Any, expected_output: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a dataset item in LangFuse
        
        Args:
            dataset_name: Name of the dataset
            input_data: Input data for the item
            expected_output: Expected output for the item
            metadata: Optional metadata
            
        Returns:
            Dict containing the created item information
        """
        request_id = get_request_id()
        
        try:
            payload = {
                "datasetName": dataset_name,
                "input": input_data,
                "expectedOutput": expected_output
            }
            
            if metadata:
                payload["metadata"] = metadata
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/public/dataset-items",
                    json=payload,
                    auth=(self.public_key, self.secret_key),
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if response.status_code in [200, 201]:
                    return response.json()
                else:
                    error_detail = response.text
                    raise DatasetError(
                        f"Failed to create dataset item: {response.status_code} - {error_detail}",
                        request_id=request_id,
                        dataset_name=dataset_name
                    )
                    
        except httpx.HTTPError as e:
            raise DatasetError(
                f"HTTP error while creating dataset item: {str(e)}",
                request_id=request_id,
                dataset_name=dataset_name
            )
    
    async def upload_dataset_items_batch(self, dataset_name: str, items: List[Dict[str, Any]], 
                                       max_concurrent: int = 3, retry_attempts: int = 3) -> Dict[str, Any]:
        """
        Upload multiple dataset items in batch with async concurrency and rate limiting
        
        Args:
            dataset_name: Name of the dataset
            items: List of items to upload
            max_concurrent: Maximum concurrent requests (default: 3, safe for free tier)
            retry_attempts: Number of retry attempts for rate limits (default: 3)
            
        Returns:
            Dict containing batch upload results
        """
        request_id = get_request_id()
        
        try:
            results = {
                "total_items": len(items),
                "successful": 0,
                "failed": 0,
                "errors": [],
                "rate_limited": 0,
                "retries": 0
            }
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Progress tracking
            completed = 0
            start_time = time.time()
            
            async def upload_single_item(item_index: int, item: Dict[str, Any]) -> Dict[str, Any]:
                """Upload a single item with rate limiting and retries"""
                async with semaphore:
                    for attempt in range(retry_attempts + 1):
                        try:
                            await self.create_dataset_item(
                                dataset_name=dataset_name,
                                input_data=item.get("input"),
                                expected_output=item.get("expected_output"),
                                metadata=item.get("metadata")
                            )
                            return {"status": "success", "item_index": item_index}
                            
                        except DatasetError as e:
                            if "429" in str(e) and attempt < retry_attempts:
                                # Rate limited - exponential backoff
                                wait_time = (2 ** attempt) + (0.1 * item_index % 10)  # Add jitter
                                print(f"🔄 Rate limited on item {item_index}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{retry_attempts + 1})")
                                await asyncio.sleep(wait_time)
                                results["retries"] += 1
                                continue
                            elif "429" in str(e):
                                results["rate_limited"] += 1
                                return {"status": "rate_limited", "item_index": item_index, "error": str(e)}
                            else:
                                return {"status": "failed", "item_index": item_index, "error": str(e)}
                        except Exception as e:
                            if attempt < retry_attempts:
                                wait_time = (2 ** attempt) + (0.1 * item_index % 10)
                                print(f"🔄 Error on item {item_index}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{retry_attempts + 1})")
                                await asyncio.sleep(wait_time)
                                results["retries"] += 1
                                continue
                            else:
                                return {"status": "failed", "item_index": item_index, "error": str(e)}
                    
                    return {"status": "failed", "item_index": item_index, "error": "Max retries exceeded"}
            
            # Create tasks for all items
            tasks = [upload_single_item(i, item) for i, item in enumerate(items)]
            
            # Process tasks in chunks to avoid overwhelming the system
            chunk_size = min(50, len(tasks))  # Process in chunks of 50
            
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i:i + chunk_size]
                chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
                
                for result in chunk_results:
                    if isinstance(result, Exception):
                        results["failed"] += 1
                        results["errors"].append({
                            "item_index": completed,
                            "error": str(result)
                        })
                    elif result["status"] == "success":
                        results["successful"] += 1
                    elif result["status"] == "rate_limited":
                        results["failed"] += 1
                        results["errors"].append({
                            "item_index": result["item_index"],
                            "error": result["error"]
                        })
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "item_index": result["item_index"],
                            "error": result["error"]
                        })
                    
                    completed += 1
                
                # Progress reporting
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(items) - completed) / rate if rate > 0 else 0
                
                print(f"📤 Uploaded {completed}/{len(items)} items to dataset '{dataset_name}' "
                      f"({rate:.1f} items/sec, ETA: {eta:.0f}s)")
                
                # Adaptive delay between chunks based on rate limiting
                if i + chunk_size < len(tasks):
                    # Base delay of 0.5s, increase if we're seeing rate limits
                    rate_limit_ratio = results["rate_limited"] / max(1, completed)
                    adaptive_delay = 0.5 + (rate_limit_ratio * 2.0)  # Up to 2.5s delay if lots of rate limits
                    await asyncio.sleep(min(adaptive_delay, 3.0))  # Cap at 3s
            
            elapsed = time.time() - start_time
            final_rate = results["successful"] / elapsed if elapsed > 0 else 0
            
            print(f"✅ Batch upload completed in {elapsed:.1f}s ({final_rate:.1f} items/sec):")
            print(f"   - Successful: {results['successful']}")
            print(f"   - Failed: {results['failed']}")
            print(f"   - Rate limited: {results['rate_limited']}")
            print(f"   - Total retries: {results['retries']}")
            
            return results
            
        except Exception as e:
            raise DatasetError(
                f"Batch upload failed: {str(e)}",
                request_id=request_id,
                dataset_name=dataset_name
            )
    
    def parse_csv_file(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Parse CSV file content into dataset items
        
        Args:
            file_content: Raw CSV file content
            
        Returns:
            List of parsed dataset items
        """
        try:
            # Decode the file content
            content_str = file_content.decode('utf-8')
            
            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(content_str))
            items = []
            
            for row in csv_reader:
                # Assume CSV has 'input' and 'expected_output' columns
                # You can customize this based on your CSV format
                item = {
                    "input": row.get("input", ""),
                    "expected_output": self._parse_json_field(row.get("expected_output", "")),
                    "metadata": {
                        "uploaded_at": datetime.now().isoformat(),
                        "source": "csv_upload"
                    }
                }
                
                # Add any additional columns as metadata
                for key, value in row.items():
                    if key not in ["input", "expected_output"] and value:
                        item["metadata"][key] = value
                
                items.append(item)
            
            return items
            
        except Exception as e:
            raise DatasetError(f"Failed to parse CSV file: {str(e)}")
    
    def _parse_json_field(self, field_value: str) -> Any:
        """
        Try to parse a field as JSON, return as string if it fails
        """
        if not field_value:
            return ""
            
        try:
            return json.loads(field_value)
        except json.JSONDecodeError:
            return field_value 