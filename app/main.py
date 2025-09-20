"""
Main FastAPI application for Smart Health Surveillance System
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.api.routes import prediction, data_ingestion, health
from app.core.database import init_db
from app.core.config import Config

# Global variable to store ML models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting up Smart Health Surveillance System...")
    await init_db()
    
    # Load ML models
    from app.ml.model_manager import ModelManager
    model_manager = ModelManager()
    global ml_models
    ml_models = await model_manager.load_all_models()
    app.state.ml_models = ml_models
    
    yield
    
    # Shutdown
    print("Shutting down Smart Health Surveillance System...")

# Create FastAPI app
app = FastAPI(
    title="Smart Health Surveillance API",
    description="API for predicting water-borne disease outbreaks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(data_ingestion.router, prefix="/api/v1/data", tags=["data-ingestion"])
app.include_router(prediction.router, prefix="/api/v1/predict", tags=["prediction"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Health Surveillance and Early Warning System",
        "version": "1.0.0",
        "status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_DEBUG
    )
