#!/usr/bin/env python3
"""
Environment Setup Script for Complete Optimization System
Helps users configure required environment variables
"""

import os
import sys
from pathlib import Path


def check_env_var(var_name, description, required=True):
    """Check if environment variable is set"""
    value = os.getenv(var_name)
    status = "✅" if value else ("❌" if required else "⚠️")
    print(f"   {status} {var_name}: {description}")
    if value:
        print(f"      Current: {value[:20]}{'...' if len(value) > 20 else ''}")
    else:
        if required:
            print(f"      Status: MISSING (Required)")
        else:
            print(f"      Status: Not set (Optional)")
    return bool(value)


def create_env_template():
    """Create a .env template file"""
    template_content = """# Complete Optimization System Environment Variables
# Copy this file to .env and fill in your actual API keys

# Required: Anthropic Claude API key for intent analysis and optimization
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Required: Groq API key for model inference
groq_api_key=your-groq-api-key-here

# Required: LangFuse credentials for data and human feedback
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: LangFuse Project ID (auto-detected if not provided)
LANGFUSE_PROJECT_ID=your-project-id

# Optional: MongoDB connection (defaults to localhost)
MONGODB_URI=mongodb://localhost:27017/
"""
    
    env_file = Path("complete_optimization_system/.env.template")
    with open(env_file, 'w') as f:
        f.write(template_content)
    
    print(f"📝 Created environment template: {env_file}")
    return env_file


def main():
    """Main setup function"""
    print("🔧 Complete Optimization System - Environment Setup")
    print("=" * 60)
    
    print("\n📋 Checking Environment Variables:")
    print("-" * 40)
    
    # Check required variables
    required_vars = [
        ("ANTHROPIC_API_KEY", "Anthropic Claude API key for optimization"),
        ("groq_api_key", "Groq API key for model inference"),
        ("LANGFUSE_SECRET_KEY", "LangFuse secret key for data access"),
        ("LANGFUSE_PUBLIC_KEY", "LangFuse public key for data access"),
    ]
    
    optional_vars = [
        ("LANGFUSE_HOST", "LangFuse host URL (defaults to cloud.langfuse.com)"),
        ("LANGFUSE_PROJECT_ID", "LangFuse project ID (auto-detected if not provided)"),
        ("MONGODB_URI", "MongoDB connection URI (defaults to localhost)"),
    ]
    
    missing_required = []
    
    for var_name, description in required_vars:
        if not check_env_var(var_name, description, required=True):
            missing_required.append(var_name)
    
    print("\n📋 Optional Variables:")
    print("-" * 40)
    
    for var_name, description in optional_vars:
        check_env_var(var_name, description, required=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    if not missing_required:
        print("🎉 All required environment variables are set!")
        print("✅ System is ready to run.")
        
        print("\n🚀 Next steps:")
        print("   1. Test the system: python complete_optimization_system/test_system.py")
        print("   2. Run optimization: python complete_optimization_system/run.py")
        
    else:
        print(f"❌ Missing {len(missing_required)} required environment variables:")
        for var in missing_required:
            print(f"   • {var}")
        
        print("\n💡 How to set environment variables:")
        print("   Option 1 - Export in terminal:")
        for var in missing_required:
            print(f"   export {var}='your-api-key-here'")
        
        print("\n   Option 2 - Create .env file:")
        env_template = create_env_template()
        print(f"   1. Edit {env_template}")
        print("   2. Rename to .env")
        print("   3. Load with: source .env")
        
        print("\n📚 Where to get API keys:")
        print("   • Anthropic Claude: https://console.anthropic.com/")
        print("   • Groq: https://console.groq.com/")
        print("   • LangFuse: https://cloud.langfuse.com/")
    
    print("\n" + "=" * 60)
    
    return len(missing_required) == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1) 