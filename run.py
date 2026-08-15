"""
YOLO12 交通标志检测系统 — 一步启动
====================================
自动构建前端（如需要）并启动 API 服务。

使用方法:
    python run.py            # 启动服务
    python run.py --build    # 先构建前端再启动
    python run.py --port 8080  # 自定义端口
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
DIST_INDEX = DIST_DIR / "index.html"

# ── 终端颜色（Windows 兼容） ──────────────────────────

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    os.system("")  # 启用 Windows 终端 ANSI 颜色支持

C_RESET = "\033[0m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"


def _print_header(title: str) -> None:
    """打印标题横幅"""
    bar = "=" * 56
    print(f"\n{C_CYAN}{bar}")
    print(f"  {title}")
    print(f"{bar}{C_RESET}\n")


def _ok(msg: str) -> None:
    print(f"{C_GREEN}✅ {msg}{C_RESET}")


def _warn(msg: str) -> None:
    print(f"{C_YELLOW}⚠️  {msg}{C_RESET}")


def _err(msg: str) -> None:
    print(f"{C_RED}❌ {msg}{C_RESET}")


def _info(msg: str) -> None:
    print(f"{C_CYAN}ℹ️  {msg}{C_RESET}")


# ── 前端构建 ──────────────────────────────────────────

def check_frontend_built() -> bool:
    """检查前端是否已构建（dist/ 中存在 index.html）"""
    return DIST_INDEX.exists()


def build_frontend() -> bool:
    """
    构建前端：npm install → npm run build。
    返回 True 表示构建成功。
    """
    if not FRONTEND_DIR.exists():
        _warn("前端目录（frontend/）不存在，跳过前端构建。")
        return False

    if not (FRONTEND_DIR / "package.json").exists():
        _warn("未找到 frontend/package.json，跳过前端构建。")
        return False

    # npm install
    _info("安装前端依赖 (npm install)...")
    result = subprocess.run(
        ["npm", "install"],
        cwd=str(FRONTEND_DIR),
    )
    if result.returncode != 0:
        _err("npm install 失败。请确保已安装 Node.js（https://nodejs.org/）。")
        return False

    # npm run build
    _info("构建前端 (npm run build)...")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_DIR),
    )
    if result.returncode != 0:
        _err("前端构建失败，请检查 frontend/ 源码是否有问题。")
        return False

    _ok("前端构建完成")
    return True


# ── 主入口 ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO12 交通标志检测系统 — 一步启动 API + 前端",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="强制重新构建前端（npm install && npm run build）",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API 服务端口（默认: 8000）",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="API 服务监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--no-frontend", action="store_true",
        help="跳过前端构建检查，纯 API 模式启动",
    )
    args = parser.parse_args()

    # ── 设置环境变量（demo 的 ServiceConfig 会读取） ─
    os.environ.setdefault("API_HOST", args.host)
    os.environ.setdefault("API_PORT", str(args.port))

    # ── 前端构建 ───────────────────────────────────────
    if not args.no_frontend:
        if args.build:
            _print_header("前端构建（--build 模式）")
            ok = build_frontend()
            if not ok:
                _warn("前端构建失败，将以纯 API 模式启动。")
        elif not check_frontend_built():
            _info("前端尚未构建，开始自动构建...")
            ok = build_frontend()
            if not ok:
                _warn("前端构建失败，将以纯 API 模式启动。")

    # ── 启动信息 ───────────────────────────────────────
    _print_header("YOLO12 交通标志检测系统")
    _info(f"API 地址:   http://{args.host}:{args.port}")
    _info(f"API 文档:   http://{args.host}:{args.port}/api/docs")
    if check_frontend_built():
        _info(f"前端页面:   http://{args.host}:{args.port}/")
        print()
        _ok("已就绪，浏览器访问上方地址即可使用。")
    else:
        print()
        _info("纯 API 模式运行中，访问 /api/docs 查看接口文档。")
    print()

    # ── 启动后端 ───────────────────────────────────────
    try:
        import uvicorn
    except ImportError:
        _err("未安装 uvicorn。请执行: pip install uvicorn[standard]")
        sys.exit(1)

    try:
        uvicorn.run(
            "demo.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}服务已停止。{C_RESET}")
    except Exception as exc:
        _err(f"服务启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
