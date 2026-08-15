"""
YOLO12 交通标志检测系统 — 统一推理 API
=======================================
FastAPI 应用入口。
"""
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from demo.config import ServiceConfig
from inference.config import ROOT
from demo.routers import detect, models, stream

# ── 创建应用 ─────────────────────────────────────────

app = FastAPI(
    title="YOLO12 交通标志检测系统",
    description="统一推理底座 + 演示系统，支持图片/视频/实时流三种检测方式",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── 可选：挂载前端静态文件 ──────────────────────────

FRONTEND_DIST = ROOT / "frontend" / "dist"
_HAS_FRONTEND = FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists()

if _HAS_FRONTEND:
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend_assets",
    )

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """提供前端页面"""
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/detect/{path:path}", include_in_schema=False)
    async def serve_frontend_routes(path: str):
        """SPA 路由：所有 /detect/* 路径返回 index.html"""
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    print(f"✅ 前端已挂载: http://localhost:{ServiceConfig.PORT}/")
else:
    print(f"ℹ️  前端静态文件未构建，仅提供 API 服务")
    print(f"   构建方法: cd frontend && npm run build")

# ── CORS ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ─────────────────────────────────────────

app.include_router(detect.router)
app.include_router(models.router)
app.include_router(stream.router)


# ── 健康检查 ─────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "ok",
        "service": "yolo12-traffic-sign-detection",
        "version": "1.0.0",
    }


# ── 全局异常处理 ─────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理的异常"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "服务器内部错误",
            "detail": str(exc) if ServiceConfig.RELOAD else "请联系管理员",
        },
    )


# ── 请求耗时中间件 ───────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = str(round(elapsed, 1))
    return response
