"""Edge AI CCTV Surveillance Core - FastAPI Application Entrypoint."""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.config import settings
from app.database import engine, async_session_factory
from app.models.db_models import Base, CameraModel
from app.routes import cameras, events, webrtc, health, dvr, zones, setup
from app.services.video_ingest_service import video_ingest_service
from app.services.clip_recorder import StorageCleaner
from app.services.dvr_recorder import dvr_recorder_service
from app.services.mdns_service import mdns_advertiser
from app.services.resilience import setup_structured_logging, ServiceHealthTracker

# Setup structured JSON logging with rotating file handler
setup_structured_logging()
logger = logging.getLogger("CCTVCore")

DEFAULT_CAMERAS = [
    {
        "id": "cam_living_room",
        "name": "Living Room",
        "location": "Ground Floor",
        "rtsp_url": "rtsp://192.168.1.50:554/stream1",
        "webrtc_url": f"{settings.EDGE_BASE_URL}/api/v1/webrtc/offer?camera_id=cam_living_room",
        "status": "ONLINE",
        "fps": 30,
        "resolution": "1920x1080",
        "is_ai_enabled": True,
        "ai_models": ["yolov8n", "yolov8n_pose"],
        "dvr_enabled": True,
        "dvr_retention_days": 7,
        "dvr_quota_gb": 100.0
    },
    {
        "id": "cam_front_door",
        "name": "Front Door Entrance",
        "location": "Exterior Front",
        "rtsp_url": "rtsp://192.168.1.51:554/stream1",
        "webrtc_url": f"{settings.EDGE_BASE_URL}/api/v1/webrtc/offer?camera_id=cam_front_door",
        "status": "ONLINE",
        "fps": 30,
        "resolution": "1920x1080",
        "is_ai_enabled": True,
        "ai_models": ["yolov8n"],
        "dvr_enabled": True,
        "dvr_retention_days": 7,
        "dvr_quota_gb": 100.0
    },
    {
        "id": "cam_backyard",
        "name": "Backyard & Patio",
        "location": "Exterior Rear",
        "rtsp_url": "rtsp://192.168.1.52:554/stream1",
        "webrtc_url": f"{settings.EDGE_BASE_URL}/api/v1/webrtc/offer?camera_id=cam_backyard",
        "status": "ONLINE",
        "fps": 25,
        "resolution": "1920x1080",
        "is_ai_enabled": True,
        "ai_models": ["yolov8n"],
        "dvr_enabled": True,
        "dvr_retention_days": 7,
        "dvr_quota_gb": 100.0
    },
]


async def periodic_dvr_indexer_and_cleaner():
    """Background task running every 30s to index new MP4 segments and enforce retention/quotas."""
    while True:
        try:
            await dvr_recorder_service.scan_and_index_segments()
            # Offload synchronous disk IO to worker thread
            await asyncio.to_thread(
                StorageCleaner.cleanup_old_media,
                storage_dir=settings.CLIPS_DIR,
                max_age_days=settings.STORAGE_RETENTION_DAYS,
                max_disk_percent=settings.STORAGE_MAX_DISK_PERCENT
            )
            await dvr_recorder_service.enforce_retention_and_quotas()
        except Exception as e:
            logger.error(f"DVR background cleaner error: {e}")
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Initialize SQLite schema, seed cameras, start workers, mDNS & DVR."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Hardware Passthrough: VA-API={settings.VAAPI_DEVICE}, HailoRT={settings.HAILO_DEVICE}")
    
    # Initialize Service Health Tracker
    health_tracker = ServiceHealthTracker()

    # 1. Initialize SQLite Database Schema
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        health_tracker.record_success("database")
    except Exception as e:
        logger.error(f"Database schema initialization failed: {e}")
        health_tracker.record_failure("database", str(e))
        
    # Check if setup is completed
    try:
        from app.services.setup_service import setup_service
        async with async_session_factory() as session:
            is_setup = await setup_service.is_setup_completed(session)
            app.state.setup_completed = is_setup
            if not is_setup:
                logger.warning(
                    "\n=======================================================\n"
                    " SYSTEM SETUP REQUIRED!\n"
                    " Please visit /api/v1/setup/status to complete setup.\n"
                    "=======================================================\n"
                )
    except Exception as e:
        logger.error(f"Failed to check setup status: {e}")

    # 2. Seed default cameras if table is empty
    try:
        async with async_session_factory() as session:
            stmt = select(CameraModel)
            res = await session.execute(stmt)
            existing = res.scalars().all()
            if not existing:
                logger.info("Seeding default camera configurations into SQLite DB...")
                for c_data in DEFAULT_CAMERAS:
                    session.add(CameraModel(**c_data))
                await session.commit()
    except Exception as e:
        logger.error(f"Failed to seed default cameras: {e}")

    # 3. Start AI Video Ingest & 24/7 Segmented DVR Workers
    try:
        async with async_session_factory() as session:
            res = await session.execute(select(CameraModel))
            cameras_list = res.scalars().all()
            for camera in cameras_list:
                try:
                    await video_ingest_service.register_and_start_camera(camera.id, camera.rtsp_url)
                    if camera.dvr_enabled:
                        dvr_recorder_service.start_camera_dvr(camera.id, camera.rtsp_url)
                    health_tracker.record_success(f"rtsp_{camera.id}")
                except Exception as e:
                    logger.warning(f"Could not auto-start ingest/DVR for {camera.id}: {e}")
                    health_tracker.record_failure(f"rtsp_{camera.id}", str(e))
    except Exception as e:
        logger.error(f"Failed to load cameras for auto-start: {e}")

    # 4. Start mDNS Zeroconf Broadcaster for local Flutter app discovery
    try:
        await mdns_advertiser.start()
    except Exception as e:
        logger.warning(f"mDNS advertiser failed to start: {e}")

    # 5. Start background DVR indexer and storage cleaner
    cleaner_task = None
    try:
        cleaner_task = asyncio.create_task(periodic_dvr_indexer_and_cleaner())
    except Exception as e:
        logger.warning(f"DVR background cleaner failed to start: {e}")

    yield

    logger.info("Shutting down Edge CCTV Core services...")
    if cleaner_task:
        cleaner_task.cancel()
        try:
            await cleaner_task
        except asyncio.CancelledError:
            pass
    try:
        await mdns_advertiser.stop()
    except Exception as e:
        logger.error(f"Error stopping mDNS: {e}")
    try:
        dvr_recorder_service.stop_all()
    except Exception as e:
        logger.error(f"Error stopping DVR: {e}")
    try:
        await video_ingest_service.stop_all()
    except Exception as e:
        logger.error(f"Error stopping video ingest: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="100% Edge-Processed CCTV AI Surveillance Backend with HailoRT, Intel QuickSync, 24/7 DVR & WebRTC.",
    lifespan=lifespan
)

if "*" in settings.ALLOWED_CORS_ORIGINS:
    allow_origins = []
    allow_origin_regex = ".*"
else:
    allow_origins = settings.ALLOWED_CORS_ORIGINS
    allow_origin_regex = None

# CORS middleware with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Edge-API-Key"],
)

import time
from fastapi.responses import JSONResponse
from fastapi import Request

@app.middleware("http")
async def setup_check_middleware(request: Request, call_next):
    # Bypass for setup routes, auth routes, health check, docs, root, and CORS OPTIONS preflights
    path = request.url.path
    if (
        request.method == "OPTIONS"
        or path.startswith("/api/v1/setup")
        or path.startswith("/api/v1/auth")
        or path.startswith("/api/v1/health")
        or path in ("/", "/docs", "/openapi.json", "/favicon.ico")
    ):
        return await call_next(request)
        
    # Check if setup is completed (caching the result to avoid DB hits every request would be better in prod)
    if getattr(app.state, "setup_completed", False) is False:
        # Check DB
        try:
            from app.services.setup_service import setup_service
            async with async_session_factory() as session:
                is_completed = await setup_service.is_setup_completed(session)
                app.state.setup_completed = is_completed
                if not is_completed:
                    return JSONResponse(
                        status_code=403,
                        content={"error": "setup_required", "setup_url": "/api/v1/setup/status"}
                    )
        except Exception:
            pass
            
    return await call_next(request)

@app.middleware("http")
async def add_process_time_header_and_log(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": process_time_ms
            }
        )
        return response
    except Exception as e:
        process_time_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Uncaught exception: {str(e)}",
            exc_info=True,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": process_time_ms,
                "error_type": type(e).__name__
            }
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error_type": type(e).__name__}
        )

# Mount Routers
app.include_router(setup.router, prefix="/api/v1")
app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(webrtc.router)
app.include_router(dvr.router)
app.include_router(zones.router)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "online",
        "health_url": "/api/v1/health",
        "docs_url": "/docs"
    }
