"""
环境验证脚本 - 确认 YOLO12 项目运行环境就绪
在 PyCharm 中直接右键运行即可。

检查内容：
  1. Python 版本
  2. PyTorch 版本与设备后端（CUDA / MPS / CPU）
  3. GPU 算力验证（多卡遍历，真实张量运算测试）
  4. ultralytics 是否可导入
  5. YOLO12 预训练权重能否下载（注意: 官方命名为 yolo12，非 yolov12）
"""
import sys
import platform


def _get_architecture(cap: tuple) -> str:
    """根据 compute capability 返回 NVIDIA GPU 架构名称。"""
    major, minor = cap
    if major == 2:
        return "Fermi"
    if major == 3:
        return "Kepler"
    if major == 5:
        return "Maxwell"
    if major == 6:
        return "Pascal"
    if major == 7:
        return "Volta" if minor < 5 else "Turing"
    if major == 8:
        return "Ampere"
    if major == 9:
        return "Ada Lovelace"
    if major >= 12:
        return "Blackwell"
    return f"Unknown (sm_{major}{minor})"


def check_python():
    print("=" * 60)
    print("[1/5] 检查 Python 环境")
    print(f"  Python 版本: {platform.python_version()}")
    print(f"  解释器路径: {sys.executable}")
    print(f"  平台架构: {platform.architecture()[0]}")
    # 建议使用 3.8-3.11，规避 3.13 的部分依赖兼容问题
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 8)
    print(f"  状态: {'OK' if ok else 'WARN - 建议 Python 3.8-3.11'}")


def check_torch():
    """检测 PyTorch 版本及可用设备后端。"""
    print("=" * 60)
    print("[2/5] 检查 PyTorch 与设备后端")
    import torch

    print(f"  torch 版本: {torch.__version__}")
    cuda_ver = torch.version.cuda
    if cuda_ver:
        print(f"  内置 CUDA 版本: {cuda_ver}")

    # 检测所有可用后端
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False

    if cuda_ok:
        print(f"  CUDA 可用: True  (GPU 数: {torch.cuda.device_count()})")
    elif mps_ok:
        print(f"  CUDA 可用: False")
        print(f"  MPS 可用: True  (Apple Silicon GPU)")
    else:
        print(f"  CUDA 可用: False")
        print(f"  MPS 可用: False")
        print(f"  状态: WARN - 无可用的 GPU 加速后端 (将使用 CPU)")

    return {"cuda": cuda_ok, "mps": mps_ok}


def check_gpu(devices: dict):
    """验证每张 GPU 算子是否真正可用（真实张量运算）。
    光看 cuda.is_available() 不够——某些情况下能识别 GPU 但缺对应架构算子，
    实际运算会报 'no kernel image is available'。这里对每张卡跑一次矩阵运算验证。"""
    print("=" * 60)
    print("[3/5] 检查 GPU 算子可用性")

    import torch

    if not devices.get("cuda"):
        # 无 CUDA GPU，检查是否有 MPS 或只能 CPU
        if devices.get("mps"):
            print("  设备: Apple Silicon (MPS)")
            # MPS 简单张量运算验证
            try:
                a = torch.randn(500, 500, device="mps")
                b = torch.randn(500, 500, device="mps")
                c = a @ b
                print("  MPS 矩阵乘法测试: 通过")
                print("  状态: OK - MPS 可用 (Apple GPU)")
                return True
            except Exception as e:
                print(f"  状态: WARN - MPS 运算失败: {e}")
                print("  将回退到 CPU 模式")
                return False
        else:
            print("  设备: CPU (无可用 GPU)")
            print("  状态: WARN - 仅 CPU 可用，训练速度会很慢，但仍可运行")
            return False  # 并非致命，但不建议训练

    # ── CUDA GPU 多卡遍历 ──
    num_gpus = torch.cuda.device_count()
    all_ok = True
    needs_newer_torch = False
    suggested_cuda_ver = torch.version.cuda or "?"

    for i in range(num_gpus):
        print(f"  --- GPU {i} ---")
        try:
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            arch = _get_architecture(cap)
            print(f"  名称: {name}")
            print(f"  算力: {cap[0]}.{cap[1]}  ({arch} 架构)")

            # 张量运算验证
            with torch.cuda.device(i):
                a = torch.randn(1000, 1000, device=f"cuda:{i}")
                b = torch.randn(1000, 1000, device=f"cuda:{i}")
                c = a @ b
                torch.cuda.synchronize(i)
                mem_mb = torch.cuda.memory_allocated(i) / 1024 / 1024
                torch.cuda.empty_cache()
            print(f"  矩阵乘法测试: 通过 (均值 {c.mean().item():.4f}, 显存 {mem_mb:.1f} MB)")
            print(f"  状态: OK")
        except Exception as e:
            all_ok = False
            err_msg = str(e)
            print(f"  状态: FAIL - {err_msg}")
            if "no kernel image" in err_msg.lower() or "no kernel" in err_msg.lower():
                needs_newer_torch = True
                failed_arch = "当前"
                # 根据架构给出建议
                try:
                    cap = torch.cuda.get_device_capability(i)
                    failed_arch = _get_architecture(cap)
                    if cap[0] >= 12:
                        suggested_cuda_ver = "12.8"
                    elif cap[0] >= 9:
                        suggested_cuda_ver = "12.4"
                    elif cap[0] >= 8:
                        suggested_cuda_ver = "12.1"
                except Exception:
                    pass

    if needs_newer_torch:
        cu_tag = f"cu{suggested_cuda_ver.replace('.', '')}"
        print(f"  排查: torch 版本缺少 {failed_arch} 架构的 CUDA 算子")
        print(f"        请安装对应版本: pip install torch --index-url https://download.pytorch.org/whl/{cu_tag}")

    return all_ok


def check_ultralytics():
    print("=" * 60)
    print("[4/5] 检查 ultralytics")
    try:
        import ultralytics
        print(f"  ultralytics 版本: {ultralytics.__version__}")
        print("  状态: OK")
        return True
    except ImportError as e:
        print(f"  状态: FAIL - {e}")
        return False


def check_yolo_weights():
    print("=" * 60)
    print("[5/5] 检查 YOLO12 预训练权重下载")
    try:
        from ultralytics import YOLO
        # 注意: ultralytics 内部命名为 yolo12（不是 yolov12）！
        # 首次运行会从 GitHub v8.4.0 release 自动下载 yolo12n.pt
        model = YOLO("yolo12n.pt")
        print(f"  YOLO12n 模型加载成功")
        print("  状态: OK")
        return True
    except Exception as e:
        print(f"  状态: WARN - 权重下载/加载失败: {e}")
        print("  排查: 可能是网络问题，可手动下载 yolo12n.pt 放到项目根目录")
        print("        下载地址: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt")
        return False


if __name__ == "__main__":
    print("YOLO12 交通标志识别项目 - 环境验证")
    print("说明: 5 项全部 OK 即可开始训练\n")
    check_python()
    devices = check_torch()
    gpu_ok = check_gpu(devices)
    check_ultralytics()
    check_yolo_weights()
    print("=" * 60)
    if gpu_ok:
        print("结论: 核心环境就绪，GPU 可用，可以开始训练。")
    elif devices.get("mps"):
        print("结论: MPS 可用 (Apple GPU)，环境就绪，可以开始训练。")
    elif devices.get("cuda"):
        print("结论: GPU 算子异常，请先解决 CUDA/驱动问题再继续。")
    else:
        print("结论: 仅 CPU 可用，环境就绪但训练速度会较慢。")
