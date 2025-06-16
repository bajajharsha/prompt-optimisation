"""
Simple Optimizer Registry - Just a list of available optimizers
"""

import os
import sys

# Add project root to path - fix the import issue
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from prompt_optimizer.optimizers.freeform_optimizer import FreeformOptimizer
from prompt_optimizer.utils.claude_client import ClaudeClient


def get_available_optimizers(claude_client: ClaudeClient = None):
    """
    Simple function that returns list of available optimizers
    Just like your agent example - simple and direct
    """
    
    # Create optimizers with their configurations
    optimizers = [
        {
            "name": "freeform",
            "optimizer": FreeformOptimizer(claude_client=claude_client),
            "description": "Intelligent freeform prompt optimizer using Claude to analyze context and improve prompts"
        }
        # Add more optimizers here as we build them:
        # {
        #     "name": "few_shot",
        #     "optimizer": FewShotOptimizer(claude_client=claude_client),
        #     "description": "Few-shot example-based optimization"
        # },
        # {
        #     "name": "chain_of_thought", 
        #     "optimizer": ChainOfThoughtOptimizer(claude_client=claude_client),
        #     "description": "Chain-of-thought reasoning optimization"
        # }
    ]
    
    return optimizers


def get_optimizer_by_name(name: str, claude_client: ClaudeClient = None):
    """Get a specific optimizer by name"""
    optimizers = get_available_optimizers(claude_client)
    
    for opt_config in optimizers:
        if opt_config["name"] == name:
            return opt_config["optimizer"]
    
    return None


def get_optimizer_names():
    """Get list of available optimizer names"""
    optimizers = get_available_optimizers()
    return [opt["name"] for opt in optimizers]


def get_default_optimizer(claude_client: ClaudeClient = None):
    """Get the default optimizer (first in list)"""
    optimizers = get_available_optimizers(claude_client)
    return optimizers[0]["optimizer"] if optimizers else None


# That's it! No complex registry class, no global state, no unregister
# Just simple functions that return what we need

if __name__ == "__main__":
    print("🔧 Simple Optimizer Registry")
    print("=" * 40)
    
    # Test the simple registry
    optimizers = get_available_optimizers()
    
    print(f"Available optimizers: {len(optimizers)}")
    for opt in optimizers:
        print(f"  - {opt['name']}: {opt['description']}")
    
    # Test getting by name
    freeform = get_optimizer_by_name("freeform")
    print(f"\nGot freeform optimizer: {freeform is not None}")
    
    # Test default
    default = get_default_optimizer()
    print(f"Default optimizer: {default.name if default else 'None'}")
    
    print("\n✅ Simple registry works!") 