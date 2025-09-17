from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from vidaio_subnet_core import CONFIG
import uvicorn

from config import logger
from routes import router, admin_router
from scheduler import scheduler, setup_data_cleanup_job
from config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (previously in on_event("startup"))
    settings = get_settings()
    setup_data_cleanup_job(settings)
    logger.info(f"Data cleanup scheduled to run every {settings.CLEANUP_INTERVAL_HOURS} hours")
    logger.info(f"Data retention period set to {settings.DATA_RETENTION_DAYS} days")
    
    yield  # This is where the application runs
    
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Cleanup scheduler shut down")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Video Processing Subnet API",
    description="API for handling video upscaling and compression tasks",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(router)
app.include_router(admin_router)

# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

if __name__ == "__main__":
    host = CONFIG.organic_gateway.host
    port = CONFIG.organic_gateway.port
    uvicorn.run("server:app", host=host, port=port)
