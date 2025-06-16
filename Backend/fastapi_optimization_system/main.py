from contextlib import asynccontextmanager
import traceback
import sys
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes.optimization_routes import router as optimization_router
from app.routes.upload_dataset_routes import router as upload_dataset_router
from app.utils.context_util import request_context
from app.utils.error_handler import OptimizationError

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifecycle management"""
    print("🚀 Starting Prompt Optimization API...")
    yield
    print("🛑 Shutting down Prompt Optimization API...")

app = FastAPI(
    title="Auto Prompt Optimization API",
    description="Automated prompt optimization system for JSON responses with enum validation",
    version="1.0.0",
    lifespan=app_lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(optimization_router, prefix="/api/v1")
app.include_router(upload_dataset_router, prefix="/api/v1")

@app.exception_handler(OptimizationError)
async def optimization_error_handler(request: Request, exc: OptimizationError):
    """Handle custom optimization errors"""
    print(f"OPTIMIZATION ERROR: {exc.detail}", file=sys.stderr)
    print(f"TRACEBACK: {exc.traceback}", file=sys.stderr)
    return exc.response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    trace = traceback.format_exc()
    print(f"UNHANDLED ERROR: {str(exc)}", file=sys.stderr)
    print(f"TRACEBACK: {trace}", file=sys.stderr)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during optimization",
            "request_id": request_context.get()
        }
    )

@app.middleware("http")
async def set_request_context(request: Request, call_next):
    """Set request context for tracking"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_context.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
    finally:
        request_context.reset(token)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Auto Prompt Optimization API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 