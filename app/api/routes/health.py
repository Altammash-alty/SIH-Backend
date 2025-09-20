"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "health-surveillance"}

@router.get("/db")
async def database_health(db: AsyncSession = Depends(get_db)):
    """Database health check"""
    try:
        # Test database connection
        result = await db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    return {"status": "ready"}
