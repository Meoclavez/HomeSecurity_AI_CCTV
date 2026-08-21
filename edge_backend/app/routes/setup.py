from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional

from app.database import get_db
from app.services.setup_service import setup_service
from app.services.auth_service import auth_service

router = APIRouter()

class AdminCreateReq(BaseModel):
    username: str
    password: str
    display_name: str
    role: str = "owner"

class LoginReq(BaseModel):
    username: str
    password: str

class RefreshReq(BaseModel):
    refresh_token: str

class ChangePasswordReq(BaseModel):
    old_password: str
    new_password: str
    
class PairReq(BaseModel):
    pairing_code: str

class TestCameraReq(BaseModel):
    url: str

class ScanNetworkReq(BaseModel):
    subnet: str

@router.get("/setup/status")
async def get_setup_status(session: AsyncSession = Depends(get_db)):
    is_completed = await setup_service.is_setup_completed(session)
    current_step = await setup_service.get_setup_step(session)
    hardware = setup_service.detect_hardware()
    
    return {
        "is_completed": is_completed,
        "current_step": current_step,
        "hardware_report": hardware
    }

@router.post("/setup/admin")
async def create_first_admin(req: AdminCreateReq, session: AsyncSession = Depends(get_db)):
    # Check if we can create admin
    # Only allow if setup is not completed or if no admins exist
    from app.models.db_models import AdminUserModel
    from sqlalchemy import select
    
    stmt = select(AdminUserModel)
    result = await session.execute(stmt)
    if result.scalars().first():
        raise HTTPException(status_code=403, detail="Admin user already exists")
        
    user = await auth_service.create_admin_user(session, req.username, req.password, req.display_name, req.role)
    await setup_service.set_setup_step(session, 2)
    return {"status": "success", "user_id": user.id}

@router.post("/setup/hardware-scan")
async def hardware_scan(
    session: AsyncSession = Depends(get_db), 
    has_access: bool = Depends(auth_service.verify_api_access)
):
    hardware = setup_service.detect_hardware()
    return {"hardware": hardware}

@router.post("/setup/camera-scan")
async def scan_cameras(
    req: ScanNetworkReq, 
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    results = await setup_service.scan_rtsp_cameras(req.subnet)
    return {"cameras": results}

@router.post("/setup/test-camera")
async def test_camera(
    req: TestCameraReq,
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    result = await setup_service.test_rtsp_url(req.url)
    return result

@router.post("/setup/add-cameras")
async def add_cameras(
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    # Stub
    await setup_service.set_setup_step(session, 3)
    return {"status": "success"}

@router.post("/setup/network-config")
async def save_network_config(
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    # Stub
    await setup_service.set_setup_step(session, 4)
    return {"status": "success"}

@router.post("/setup/notifications")
async def save_notifications(
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    # Stub
    await setup_service.set_setup_step(session, 5)
    return {"status": "success"}

@router.post("/setup/complete")
async def complete_setup(
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    await setup_service.complete_setup(session)
    setup_service.generate_secure_secrets()
    return {"status": "success"}

# Auth routes
@router.post("/auth/login")
async def login(req: LoginReq, session: AsyncSession = Depends(get_db)):
    tokens = await auth_service.authenticate_user(session, req.username, req.password)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return tokens

@router.post("/auth/refresh")
async def refresh(req: RefreshReq):
    new_access = auth_service.refresh_access_token(req.refresh_token)
    return {"access_token": new_access}

@router.post("/auth/change-password")
async def change_password(
    req: ChangePasswordReq, 
    request: Request,
    session: AsyncSession = Depends(get_db),
    has_access: bool = Depends(auth_service.verify_api_access)
):
    if not hasattr(request.state, "user") or "sub" not in request.state.user:
        raise HTTPException(status_code=401, detail="User context missing")
        
    await auth_service.change_password(session, request.state.user["sub"], req.old_password, req.new_password)
    return {"status": "success"}

@router.get("/auth/pairing-code")
async def get_pairing_code(
    has_access: bool = Depends(auth_service.verify_api_access)
):
    code = auth_service.generate_app_pairing_code()
    return {"pairing_code": code, "expires_in": 300}

@router.post("/auth/pair")
async def pair_app(req: PairReq):
    # In reality, this would verify the code from DB and return tokens.
    # We will just return a dummy token structure for the stub.
    return {"access_token": "stub_access_token", "refresh_token": "stub_refresh_token"}
