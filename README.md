# YOLO12 交通标志识别项目

基于 [Ultralytics YOLO12](https://docs.ultralytics.com/models/yolo12/) 的交通标志检测系统，使用 **COCO 格式 46 类中国交通标志数据集**训练，支持图片、视频、摄像头三种推理方式，并附带完整的 Web 演示系统（图片/视频/浏览器摄像头实时检测）。

> ⚠️ **命名说明**：Ultralytics 官方将该模型命名为 **`yolo12`**（不是 `yolov12`）。模型文件为 `yolo12n.pt`。许多教程写作 "YOLOv12" 是不准确的。

---

## 一、硬件环境

本项目在以下环境验证通过：

| 组件 | 配置 |
|------|------|
| GPU | NVIDIA 显卡（Blackwell 架构，8GB 及以上显存，需 CUDA 12.8+） |
| PyTorch | 2.x+cu128（**RTX 50 系列必须 CUDA 12.8 版**） |
| Python | 3.11 |
| Node.js | 18+（构建前端，可选） |

**关键点**：RTX 50 系列是 Blackwell 架构（compute capability 12.0），必须使用 cu128 版 PyTorch，否则会报 `no kernel image is available` 错误。其他显卡使用对应 CUDA 版本的 PyTorch 即可。

---

## 二、安装步骤

### 1. 创建 conda 环境

```bash
conda create -n yolov12 python=3.11 -y
conda activate yolov12
```

### 2. 安装 PyTorch（CUDA 12.8 版，RTX 50 系列关键步骤）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 4. 验证环境

```bash
python src/verify_env.py
```

看到 5 项全部 OK 即环境就绪。特别确认 `[3/5] GPU 算子可用性` 通过。

---

## 三、项目结构

```
traffic-sign-detection/
├── inference/               # 推理底座 — 纯 YOLO，零 Web 依赖
│   ├── engine.py            # DetectorService（模型加载/图片/视频/帧推理）
│   ├── config.py            # InferenceConfig（阈值/尺寸/设备）
│   ├── schemas.py           # Detection Pydantic 数据类
│   ├── label_mapping.py     # 中文名称映射 get_chinese_name()
│   └── coco_names.json      # 46 类中英文对照表
├── demo/                    # 演示系统 — FastAPI，单向依赖 inference
│   ├── main.py              # FastAPI 入口（自动挂载 frontend/dist/）
│   ├── config.py            # ServiceConfig（host/port/路径/Redis）
│   ├── schemas.py           # API 响应模型
│   ├── routers/
│   │   ├── detect.py        # 图片/视频检测、进度轮询、结果下载
│   │   ├── models.py        # 模型列表/切换/类别名
│   │   └── stream.py        # WebSocket 实时流推理
│   └── tasks/
│       └── video_tasks.py   # Celery 异步视频任务（可选，需 Redis）
├── frontend/                # Vue 3 + Vite + Element Plus + ECharts
│   └── src/views/           # ImageDetect / VideoDetect / CameraDetect
├── src/                     # CLI 脚本（独立于 Web 系统）
│   ├── verify_env.py        # 环境验证脚本
│   ├── train.py             # 训练脚本（默认 configs/coco.yaml）
│   ├── predict_image.py     # 图片推理
│   ├── predict_video.py     # 视频推理
│   └── predict_camera.py    # 摄像头实时检测
├── configs/
│   └── coco.yaml            # 数据集配置（46 类，路径为相对路径）
├── data/
│   └── coco/                # 数据集（需自行准备，见"数据集准备"）
├── models/                  # 自定义权重
├── run.py                   # 一步启动 API + 前端
├── best.pt                  # 训练好的权重（默认推理模型）
├── yolo12n.pt               # 官方预训练权重（微调/重新训练用）
├── requirements.txt
└── README.md
```

---

## 四、使用方法

### 1. 推理（开箱即用）

仓库内置训练好的 `best.pt`，克隆后配置好环境即可直接推理：

```bash
# 图片推理
python src/predict_image.py --source 你的图片.jpg

# 视频推理
python src/predict_video.py --source 你的视频.mp4

# 摄像头实时检测（q 退出，s 保存快照）
python src/predict_camera.py
```

结果默认保存于 `runs/` 目录。Web 演示系统会自动优先加载 `best.pt`。

### 2. 训练模型

```bash
# 小试跑（3 个 epoch，快速验证流程）
python src/train.py --smoke

# 正式训练（80 个 epoch）
python src/train.py

# 自定义参数
python src/train.py --model yolo12s.pt --epochs 50 --batch 8
```

训练完成后，最佳权重在 `runs/train-*/weights/best.pt`（推理服务会自动发现并加载最新的 best.pt）。

---

## 五、数据集准备

> 数据集体积较大，**不包含在仓库中**，需要自行准备。训练与推理目录约定如下：

```
data/coco/
├── images/
│   ├── train/    # 训练集图片（jpg）
│   └── val/      # 验证集图片（jpg）
└── labels/
    ├── train/    # 训练集标注（YOLO 格式 txt，与图片同名）
    └── val/      # 验证集标注（YOLO 格式 txt，与图片同名）
```

标注为 YOLO 格式：每行 `class_id x_center y_center width height`（归一化坐标），例如：

```
45 0.534722 0.453704 0.182292 0.177469
```

类别 ID 与 `configs/coco.yaml` 中的 46 类顺序一致（ID 0-45），完整对照表见文件末尾。

**数据来源参考**：本项目基于 [TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)（清华-腾讯公开交通标志数据集）整理扩展，请遵守原数据集许可协议。也可以使用 [Roboflow](https://universe.roboflow.com/) 等平台获取或自行标注数据。

**配置说明**：`configs/coco.yaml` 使用相对路径 `../data/coco`，克隆仓库并放入数据后即可直接训练，无需修改配置。

---

## 六、Web 演示系统

项目包含一个完整的 Web 演示系统，支持图片/视频/实时摄像头三种检测方式。

### 6.1 快速启动

```bash
# 一步启动（API + 前端页面）
python run.py

# 端口被占用时换端口
python run.py --port 18000

# 如需重新构建前端
python run.py --build

# 纯 API 模式（跳过前端检查）
python run.py --no-frontend
```

然后浏览器打开 `http://localhost:8000/` 即可。

### 6.2 功能说明

| 页面 | 功能 |
|------|------|
| **图片检测** | 拖拽/点击上传图片，显示原图与检测结果对比图，右侧列出检测到的交通标志和置信度 |
| **视频检测** | 上传视频文件，后台线程异步处理（`/video/process` + 进度轮询，无需 Redis），生成标注后的结果视频 |
| **实时摄像头** | 调用浏览器摄像头，通过 WebSocket 实时传输画面进行检测，实时显示检测框 |

### 6.3 API 文档

启动服务后访问 `http://localhost:8000/api/docs` 查看交互式 API 文档，支持在线测试。

### 6.4 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Vue 3 + Vite + Element Plus + ECharts |
| 后端 | FastAPI + Uvicorn |
| 推理引擎 | Ultralytics YOLO12（`inference/` 底座，`demo/` 单向依赖） |
| 视频任务 | 前端使用线程版进度接口（无需 Redis）；Celery 版（`/video/async`）可选，需 Redis |
| WebSocket | 实时流推理 |

---

## 七、常见问题

### Q1: 报错 `no kernel image is available for execution`
**原因**：PyTorch 不支持你的 GPU 架构（Blackwell）。
**解决**：重装 cu128 版 PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### Q2: 下载 `yolov12n.pt` 报 404
**原因**：文件名错了。正确文件名是 `yolo12n.pt`（没有 "v"）。
**解决**：下载地址 `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt`

### Q3: 训练时报 CUDA out of memory
**原因**：显存不足（8GB）。
**解决**：降低 batch 大小：`python src/train.py --batch 8` 或 `--batch 4`

### Q4: flash-attn 安装失败
**说明**：YOLO12 在 Linux 使用 flash-attn 加速，但 Windows 无预编译版。**不要安装 flash-attn**，ultralytics 在 Windows 会自动 fallback 到标准 attention，功能不受影响。

### Q5: 摄像头打不开
**解决**：外接 USB 摄像头编号可能是 1，试试 `python src/predict_camera.py --cam 1`

### Q6: 启动服务报 `Errno 10048`（端口被占用）
**原因**：8000 端口被其他程序占用（例如 **Steam 的 steamwebhelper.exe** 会占用 localhost 8000-8009 一带端口）。
**解决**：换端口启动，如：
```bash
python run.py --port 18000
```

---

## 八、46 类交通标志列表

类别定义以 `configs/coco.yaml` 与 `inference/coco_names.json` 为准（两处一致）：

| ID | 代码 | 含义 | ID | 代码 | 含义 |
|----|------|------|----|------|------|
| 0 | pl80 | 限制速度（80公里/小时） | 23 | pn | 禁止停车 |
| 1 | p6 | 禁止非机动车进入 | 24 | w55 | 注意儿童 |
| 2 | p5 | 禁止掉头 | 25 | p26 | 禁止载货汽车驶入 |
| 3 | pm55 | 限制质量（55吨） | 26 | p13 | 路名牌 |
| 4 | pl60 | 限制速度（60公里/小时） | 27 | pr40 | 解除限制速度（40公里/小时） |
| 5 | ip | 人行横道 | 28 | pl20 | 限制速度（20公里/小时） |
| 6 | p11 | 禁止鸣喇叭 | 29 | pm30 | 限制质量（30吨） |
| 7 | i2r | 非机动车行驶 | 30 | pl40 | 限制速度（40公里/小时） |
| 8 | p23 | 禁止向左转弯 | 31 | i2 | 机动车行驶 |
| 9 | pg | 减速让行 | 32 | pl120 | 限制速度（120公里/小时） |
| 10 | il80 | 最低限速（80公里/小时） | 33 | w32 | 无人看守铁路道口 |
| 11 | ph4 | 限制高度（4米） | 34 | ph5 | 限制高度（5米） |
| 12 | i4 | 靠右侧道路行驶 | 35 | il60 | 最低限速（60公里/小时） |
| 13 | pl70 | 限制速度（70公里/小时） | 36 | w57 | 注意行人 |
| 14 | pne | 禁止驶入 | 37 | pl100 | 限制速度（100公里/小时） |
| 15 | ph4.5 | 限制高度（4.5米） | 38 | w59 | 注意合流 |
| 16 | p12 | 禁止二轮摩托车驶入 | 39 | il100 | 最低限速（100公里/小时） |
| 17 | p3 | 禁止大型客车驶入 | 40 | p19 | 禁止向右转弯 |
| 18 | pl5 | 限制速度（5公里/小时） | 41 | pm20 | 限制质量（20吨） |
| 19 | w13 | 傍山险路 | 42 | i5 | 靠左侧道路行驶 |
| 20 | i4l | 靠左侧道路行驶 | 43 | p27 | 禁止运输危险物品车辆驶入 |
| 21 | pl30 | 限制速度（30公里/小时） | 44 | pl50 | 限制速度（50公里/小时） |
| 22 | p10 | 禁止机动车驶入 | 45 | wo | 其他 |

---

## 九、许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
