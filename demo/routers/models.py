"""
模型管理路由
============
查询可用模型列表、切换当前模型。
"""
from fastapi import APIRouter, HTTPException, Depends

from demo.schemas import ModelInfo, ModelListResponse
from inference.engine import DetectorService

router = APIRouter(prefix="/api/models", tags=["模型管理"])


def get_detector() -> DetectorService:
    return DetectorService()


@router.get("", response_model=ModelListResponse)
async def list_models(detector: DetectorService = Depends(get_detector)):
    """获取可用模型列表"""
    models = detector.get_available_models()
    current_name, current_path = detector.get_current_model()
    return ModelListResponse(
        models=[ModelInfo(**m) for m in models],
        current_model=current_path,
    )


@router.post("/switch")
async def switch_model(
    model_path: str,
    detector: DetectorService = Depends(get_detector),
):
    """切换推理模型"""
    try:
        detector.load_model(model_path)
        name, path = detector.get_current_model()
        return {"success": True, "message": f"模型已切换为: {name}", "current_model": path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@router.get("/names")
async def get_class_names(detector: DetectorService = Depends(get_detector)):
    """获取当前模型支持的类别列表"""
    names = detector.get_class_names()
    return {
        "success": True,
        "total_classes": len(names),
        "class_names": {i: name for i, name in enumerate(names)},
    }
