"""
Simple global request ID for file naming
"""

import uuid

# Global request ID
_request_id = None

def initialize_request_id():
    """Initialize a new request ID"""
    global _request_id
    _request_id = str(uuid.uuid4())
    return _request_id

def get_request_id():
    """Get the current request ID, initialize if not set"""
    global _request_id
    if _request_id is None:
        _request_id = str(uuid.uuid4())
    return _request_id 