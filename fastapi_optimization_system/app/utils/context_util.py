from contextvars import ContextVar
from typing import Optional

# Context variable for tracking request ID across async operations
request_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

def get_request_id() -> Optional[str]:
    """Get the current request ID from context"""
    return request_context.get()

def set_request_id(request_id: str) -> None:
    """Set the request ID in context"""
    request_context.set(request_id) 