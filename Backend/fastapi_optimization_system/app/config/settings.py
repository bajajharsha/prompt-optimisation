import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Auto Prompt Optimization API"
    api_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Model API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    
    # LangFuse Configuration
    langfuse_secret_key: Optional[str] = Field(default=None, env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(default=None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # MongoDB Configuration (if needed)
    mongodb_url: str = Field(default="mongodb://localhost:27017/", env="MONGODB_URL")
    mongodb_db_name: str = Field(default="prompt_optimization", env="MONGODB_DB_NAME")
    
    # Optimization Settings
    max_optimization_time: int = Field(default=3600, env="MAX_OPTIMIZATION_TIME")  # 1 hour
    default_max_iterations: int = Field(default=5, env="DEFAULT_MAX_ITERATIONS")
    default_improvement_threshold: float = Field(default=0.05, env="DEFAULT_IMPROVEMENT_THRESHOLD")
    
    # File Storage
    temp_dir: str = Field(default="/tmp/prompt_optimization", env="TEMP_DIR")
    results_dir: str = Field(default="./optimization_results", env="RESULTS_DIR")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings 