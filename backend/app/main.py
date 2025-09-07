# backend\app\main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# CORS middleware - Updated to include production URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://pdf-construction-tool-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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