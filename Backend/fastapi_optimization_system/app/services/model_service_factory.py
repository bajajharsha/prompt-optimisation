"""
Model Service Factory - Returns appropriate service based on model configuration
"""

from typing import List

from app.models.optimization_models import ModelConfiguration
from app.services.gemini_service import GeminiAPIService
from app.utils.error_handler import OptimizationError


class ModelServiceFactory:
    """Factory to get the appropriate model service based on configuration"""
    
    def __init__(self):
        # Services will be created dynamically when needed
        pass
    
    def get_service(self, model_config: ModelConfiguration):
        """Get the appropriate service based on model configuration"""
        provider = model_config.provider.value.lower()
        
        try:
            if provider == "openai":
                from app.services.openai_service import get_openai_service
                return get_openai_service()
            elif provider == "google":
                from app.services.gemini_service import get_gemini_service
                return get_gemini_service()
            elif provider == "groq":
                # Import groq service dynamically since it might not be in the main services directory
                try:
                    from app.services.groq_service import get_groq_service
                    return get_groq_service()
                except ImportError:
                    raise OptimizationError(
                        detail="Groq service not available",
                        status_code=500
                    )
            elif provider == "anthropic":
                # Import anthropic service dynamically
                try:
                    from app.services.anthropic_service import get_anthropic_service
                    return get_anthropic_service()
                except ImportError:
                    raise OptimizationError(
                        detail="Anthropic service not available",
                        status_code=500
                    )
            else:
                raise OptimizationError(
                    detail=f"Unsupported model provider: {provider}. Supported providers: groq, openai, google, anthropic",
                    status_code=400
                )
        except ValueError as e:
            # Handle API key validation errors
            if "API key not found" in str(e):
                raise OptimizationError(
                    detail=f"Configuration error for {provider}: {str(e)}",
                    status_code=400
                )
            else:
                raise OptimizationError(
                    detail=f"Error initializing {provider} service: {str(e)}",
                    status_code=500
                )
    
    async def batch_inference(
        self,
        model_config: ModelConfiguration,
        prompts: List[str],
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> List[str]:
        """
        Run batch inference using the appropriate service
        
        Args:
            model_config: Model configuration
            prompts: List of user prompts
            system_prompt: System prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            List of model responses
        """
        provider = model_config.provider.value.lower()
        model_name = model_config.model_name
        
        try:
            if provider == "openai":
                # OpenAI service batch processing
                from app.services.openai_service import get_openai_service
                openai_service = get_openai_service()
                return await openai_service.batch_completions(
                    prompts=prompts,
                    base_prompt=system_prompt,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="batch_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
                
            elif provider == "google":
                # Use Gemini service batch processing
                from app.services.gemini_service import get_gemini_service
                gemini_service = get_gemini_service()
                return await gemini_service.batch_completions(
                    prompts=prompts,
                    base_prompt=system_prompt,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="batch_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
                
            elif provider == "groq":
                # Use Groq service
                from app.services.groq_service import get_groq_service
                groq_service = get_groq_service()
                return await groq_service.batch_completions(
                    prompts=prompts,
                    base_prompt=system_prompt,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="batch_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
                
            elif provider == "anthropic":
                # Use Anthropic service
                from app.services.anthropic_service import get_anthropic_service
                anthropic_service = get_anthropic_service()
                return await anthropic_service.batch_completions(
                    prompts=prompts,
                    base_prompt=system_prompt,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="batch_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
                
        except Exception as e:
            raise OptimizationError(
                detail=f"Error in {provider} inference: {str(e)}",
                status_code=500
            )
    
    async def system_user_inference(
        self,
        model_config: ModelConfiguration,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> List[str]:
        """
        Run inference with separate system and user prompts
        
        Args:
            model_config: Model configuration
            system_prompt: System prompt
            user_prompts: List of user prompts
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            List of model responses
        """
        provider = model_config.provider.value.lower()
        model_name = model_config.model_name
        
        try:
            if provider == "groq":
                # Use Groq service with system/user separation
                from app.services.groq_service import get_groq_service
                groq_service = get_groq_service()
                return await groq_service.inference_with_system_user_prompts(
                    system_prompt=system_prompt,
                    user_prompts=user_prompts,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="system_user_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
            elif provider == "anthropic":
                # Use Anthropic service with system/user separation
                from app.services.anthropic_service import get_anthropic_service
                anthropic_service = get_anthropic_service()
                return await anthropic_service.inference_with_system_user_prompts(
                    system_prompt=system_prompt,
                    user_prompts=user_prompts,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="system_user_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
            elif provider == "openai":
                # Use OpenAI service with system/user separation
                from app.services.openai_service import get_openai_service
                openai_service = get_openai_service()
                return await openai_service.inference_with_system_user_prompts(
                    system_prompt=system_prompt,
                    user_prompts=user_prompts,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="system_user_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
            elif provider == "google":
                # Use Gemini service with system/user separation
                from app.services.gemini_service import get_gemini_service
                gemini_service = get_gemini_service()
                return await gemini_service.inference_with_system_user_prompts(
                    system_prompt=system_prompt,
                    user_prompts=user_prompts,
                    model_name=model_name,
                    component="model_service_factory",
                    operation="system_user_inference",
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
            else:
                # For other providers, fall back to batch inference with combined prompts
                combined_prompts = [f"{user_prompt}" for user_prompt in user_prompts]
                return await self.batch_inference(
                    model_config=model_config,
                    prompts=combined_prompts,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
        except Exception as e:
            raise OptimizationError(
                detail=f"Error in {provider} system/user inference: {str(e)}",
                status_code=500
            )


def get_model_service_factory() -> ModelServiceFactory:
    """Get model service factory instance"""
    return ModelServiceFactory() 