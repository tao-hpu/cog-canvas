"""
FastAPI backend for CogCanvas Web Interface.

This backend provides RESTful APIs for:
- Chat with streaming SSE responses
- Canvas object management
- Graph visualization data
- Object retrieval

To run:
    uvicorn main:app --reload --port 3801
"""

import sys
from pathlib import Path

# Add cogcanvas to path
cogcanvas_path = str(Path(__file__).parent.parent.parent.absolute())
if cogcanvas_path not in sys.path:
    sys.path.insert(0, cogcanvas_path)

# Load .env file from project root
from dotenv import load_dotenv
env_path = Path(cogcanvas_path) / ".env"
load_dotenv(env_path)
print(f"Loaded .env from: {env_path}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import canvas, chat

# Initialize FastAPI app
app = FastAPI(
    title="CogCanvas API",
    description="Backend API for CogCanvas web interface",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3800",
        "http://127.0.0.1:3800",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(canvas.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "CogCanvas API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/chat",
            "canvas": "/api/canvas",
            "graph": "/api/canvas/graph",
            "stats": "/api/canvas/stats",
            "retrieve": "/api/canvas/retrieve",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3801,
        reload=True,
        log_level="info"
    )
