"""
Data Manager - Handles data loading and stratification
"""

import json
import random
from typing import Dict, List, Any
from langfuse import Langfuse
import os


class DataManager:
    """
    Manages data loading and stratification for the optimization system
    """
    
    def __init__(self):
        self.langfuse_client = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY', 'sk-lf-d87cc28d-5a97-4fd9-bccd-13cfbf5e6ad3'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-4e626ffa-7bcd-495b-9f4d-f2f2c5b15087'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
    
    async def prepare_data_splits(self, dataset_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load data from LangFuse and create stratified splits
        
        Args:
            dataset_name: Name of the dataset in LangFuse
            
        Returns:
            Dict with train, dev_a, dev_b, test splits
        """
        print(f"📥 Loading dataset from LangFuse: {dataset_name}")
        
        # Load all data from LangFuse
        all_data = self._load_langfuse_dataset(dataset_name)
        
        print(f"✅ Loaded {len(all_data)} samples from LangFuse")
        
        # Shuffle data for random splits
        random.shuffle(all_data)
        
        # Calculate split sizes
        total_size = len(all_data)
        train_size = int(total_size * 0.25)  # 25%
        dev_a_size = int(total_size * 0.35)  # 35%
        dev_b_size = int(total_size * 0.20)  # 20%
        test_size = total_size - train_size - dev_a_size - dev_b_size  # Remaining ~20%
        
        # Create splits
        train_data = all_data[:train_size]
        dev_a_data = all_data[train_size:train_size + dev_a_size]
        dev_b_data = all_data[train_size + dev_a_size:train_size + dev_a_size + dev_b_size]
        test_data = all_data[train_size + dev_a_size + dev_b_size:]
        
        # Verify splits
        print(f"📊 Data split verification:")
        print(f"   Train: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
        print(f"   Dev A: {len(dev_a_data)} ({len(dev_a_data)/total_size*100:.1f}%)")
        print(f"   Dev B: {len(dev_b_data)} ({len(dev_b_data)/total_size*100:.1f}%)")
        print(f"   Test: {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")
        print(f"   Total: {len(train_data) + len(dev_a_data) + len(dev_b_data) + len(test_data)}")
        
        return {
            'train': train_data,
            'dev_a': dev_a_data,
            'dev_b': dev_b_data,
            'test': test_data
        }
    
    def _load_langfuse_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load dataset from LangFuse
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of data samples
        """
        try:
            dataset = self.langfuse_client.get_dataset(dataset_name)
            
            data_samples = []
            for item in dataset.items:
                try:
                    # Parse expected output
                    if isinstance(item.expected_output, dict):
                        expected_output = item.expected_output
                    elif isinstance(item.expected_output, str):
                        expected_output = json.loads(item.expected_output)
                    else:
                        print(f"⚠️  Skipping item with unexpected expected_output type: {type(item.expected_output)}")
                        continue
                    
                    # Create data sample
                    sample = {
                        "input": item.input,
                        "expected_output": expected_output,
                        "item_id": getattr(item, 'id', None)
                    }
                    data_samples.append(sample)
                    
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    print(f"⚠️  Error parsing LangFuse item: {e}")
                    continue
            
            return data_samples
            
        except Exception as e:
            print(f"❌ Error loading dataset from LangFuse: {e}")
            raise
    
    def get_train_samples_for_context(self, train_data: List[Dict[str, Any]], num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Get a subset of training samples for context
        
        Args:
            train_data: Full training data
            num_samples: Number of samples to return
            
        Returns:
            List of training samples for context
        """
        if len(train_data) <= num_samples:
            return train_data
        
        return random.sample(train_data, num_samples)
    
    def save_data_splits(self, data_splits: Dict[str, List], output_dir: str = "complete_optimization_system/data"):
        """
        Save data splits to files for debugging/analysis
        
        Args:
            data_splits: Dictionary with data splits
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in data_splits.items():
            filepath = os.path.join(output_dir, f"{split_name}_data.json")
            with open(filepath, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"💾 Saved {split_name} data to: {filepath}") 