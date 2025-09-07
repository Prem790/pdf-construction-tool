# backend\app\main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import settings
from app.api.routes import router

# Global state for storing job results
job_storage = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    print("Checking dependencies...")
    
    # Check if we have API keys
    if not settings.GEMINI_API_KEY and not settings.AZURE_VISION_KEY:
        print("WARNING: No AI API keys found. Some features may not work.")
    
    yield
    
    # Shutdown
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-powered tool to reconstruct shuffled PDF documents",
    lifespan=lifespan
)

# Define allowed origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
    "https://pdf-construction-tool-frontend.onrender.com"
]

# Add CORS middleware with explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add manual OPTIONS handler for preflight requests
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    """Handle CORS preflight requests"""
    origin = request.headers.get("origin")
    
    # Check if origin is allowed
    if origin in origins:
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "3600",
            }
        )
    
    return JSONResponse(content={"error": "CORS not allowed"}, status_code=403)

# Add request logging middleware
@app.middleware("http")
async def log_and_handle_cors(request: Request, call_next):
    """Log requests and ensure CORS headers are present"""
    origin = request.headers.get("origin")
    print(f"Request: {request.method} {request.url} from origin: {origin}")
    
    # Handle OPTIONS requests immediately
    if request.method == "OPTIONS":
        if origin in origins:
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "3600",
                }
            )
        else:
            return JSONResponse(content={"error": "CORS not allowed"}, status_code=403)
    
    # Process the request
    response = await call_next(request)
    
    # Add CORS headers to response if origin is allowed
    if origin in origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
    
    print(f"Response: {response.status_code} with CORS headers for origin: {origin}")
    return response

# Include API routes
app.include_router(router, prefix="/api")

# Mount static files
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Make job storage accessible
app.state.job_storage = job_storage

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )