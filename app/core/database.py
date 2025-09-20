"""
Database configuration and connection management
"""
import asyncio
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from app.core.config import Config

# PostgreSQL with TimescaleDB
engine = create_async_engine(Config.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# MongoDB
mongo_client = None
mongo_db = None

# Redis
redis_client = None

async def init_db():
    """Initialize database connections"""
    global mongo_client, mongo_db, redis_client
    
    # Initialize MongoDB
    mongo_client = AsyncIOMotorClient(Config.MONGODB_URL)
    mongo_db = mongo_client.health_reports
    
    # Initialize Redis
    redis_client = redis.from_url(Config.REDIS_URL)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Get database session"""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_mongo():
    """Get MongoDB database"""
    return mongo_db

async def get_redis():
    """Get Redis client"""
    return redis_client

async def close_db():
    """Close database connections"""
    global mongo_client, redis_client
    if mongo_client:
        mongo_client.close()
    if redis_client:
        await redis_client.close()
