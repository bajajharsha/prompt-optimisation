#!/usr/bin/env python3
"""
Run Script for Complete Optimization System
Simple entry point to run the full optimization workflow
"""

import asyncio
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from complete_optimization_system.main import main


if __name__ == "__main__":
    print("🚀 Starting Complete Prompt Optimization System")
    print("=" * 80)
    print("This system will:")
    print("1. Load and stratify data from LangFuse")
    print("2. Run baseline evaluation")
    print("3. Perform intent analysis")
    print("4. Execute optimization iterations")
    print("5. Collect human feedback")
    print("6. Make final deployment decision")
    print("=" * 80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ System failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 