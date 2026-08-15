# Sprint 迭代规划

> 交通标志检测系统后续迭代方向。

---

## ✅ 已完成 (Sprint 0 — 推理底座 ↔ 演示系统剥离)

- [x] `inference/` 推理底座（纯 YOLO，零 Web 依赖：engine / config / schemas / label_mapping / coco_names.json）
- [x] `demo/` 演示系统（FastAPI，单向依赖 inference；routers: detect / models / stream）
- [x] `run.py` 统一指向 `demo.main:app`
- [x] 视频处理复用 engine（删除重复循环），`detect_frame` → `detect_image` bug 修复
- [x] 全量回归通过：图片 / 视频（线程版进度）/ WebSocket 流 / 前端挂载 / CLI 脚本
- [x] 旧 `backend/` 目录已删除

> **验证结论**：`python run.py` 一键启动可用；若报 `Errno 10048` 是端口被占（Steam 常占 8000-8009），用 `python run.py --port 18000`。Celery 异步视频（`/video/async`）需 Redis，前端用的线程版进度接口无需 Redis。

---

## 🔜 近期 (Sprint 1-2)

- [ ] `src/` CLI 脚本统一导入 `inference/` 的 `DetectorService`，消除重复代码
- [ ] `_video_progress` 内存字典迁移到 Redis（解决线程安全 + 内存泄漏）
- [ ] WebSocket 支持多房间（多客户端同时实时检测）
- [ ] 前端增加检测结果历史记录

## 📋 中期 (Sprint 3-5)

- [ ] 支持上传自定义模型权重
- [ ] 批量图片检测（一次上传多张）
- [ ] 检测结果导出（CSV / JSON）
- [ ] 视频检测支持指定帧率采样
- [ ] 前端增加模型切换 UI

## 🎯 远期

- [ ] Docker 容器化部署
- [ ] GPU 推理队列管理
- [ ] 多模型并行推理
- [ ] 用户认证与权限
- [ ] 检测结果对比（不同模型/参数）
