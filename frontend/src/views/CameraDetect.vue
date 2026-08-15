<template>
  <div class="detect-view">
    <div class="main-area">
      <div class="camera-container">
        <div class="camera-box">
          <video ref="videoRef" autoplay muted playsinline class="camera-feed" @loadedmetadata="onVideoReady" />
          <canvas ref="canvasRef" class="overlay-canvas" />
          <div v-if="!cameraActive" class="camera-placeholder">
            <el-icon :size="48"><Camera /></el-icon>
            <p style="margin-top:12px;font-size:14px">点击下方按钮启动摄像头</p>
          </div>
        </div>

        <div class="camera-info">
          <span>FPS: <strong>{{ fps }}</strong></span>
          <span>当前帧: <strong>{{ frameDetections }}</strong> 个标志</span>
          <span>累计: <strong>{{ accumulatedDetections.length }}</strong> 种标志</span>
          <el-button size="small" text @click="resetAccumulated" :disabled="accumulatedDetections.length === 0">刷新目录</el-button>
          <span class="status-dot" :class="wsConnected ? 'online' : 'offline'">{{ wsConnected ? '已连接' : '未连接' }}</span>
        </div>

        <div class="camera-actions">
          <el-button size="small" circle @click.stop="refreshCamera" @mousedown.stop title="刷新摄像头"><el-icon :size="16"><Refresh /></el-icon></el-button>
          <el-select v-if="devices.length > 1" v-model="selectedDevice" placeholder="选择摄像头" size="small" style="width:200px" @change="switchDevice">
            <el-option v-for="d in devices" :key="d.deviceId" :label="d.label" :value="d.deviceId" />
          </el-select>
          <el-button :type="cameraActive ? 'danger' : 'primary'" size="large" @click="toggleCamera">
            {{ cameraActive ? '停止摄像头' : '启动摄像头' }}
          </el-button>
          <el-button size="large" @click="captureSnapshot" :disabled="!cameraActive">截取快照</el-button>
        </div>

        <!-- 快照 -->
        <div v-if="snapshots.length > 0" class="snapshots-section">
          <div class="section-title">快照 ({{ snapshots.length }})</div>
          <el-scrollbar ref="snapScrollbar" class="snap-scrollbar" @wheel.prevent="onSnapWheel">
            <div class="snapshots-row">
              <div v-for="(snap, i) in snapshots" :key="i" class="snapshot-item">
                <img :src="snap.dataUrl" class="snapshot-img" @click="previewSnapshot(i)" />
                <div v-if="snap.labels && snap.labels.length" class="snapshot-labels">
                  <el-tag v-for="(l, j) in snap.labels" :key="j" size="small" type="info" style="margin:1px">{{ l }}</el-tag>
                </div>
                <div class="snapshot-btns">
                  <el-button size="small" circle @click.stop="previewSnapshot(i)"><el-icon :size="14"><ZoomIn /></el-icon></el-button>
                  <el-button size="small" circle @click.stop="downloadSnapshot(i)"><el-icon :size="14"><Download /></el-icon></el-button>
                  <el-button size="small" circle @click.stop="removeSnapshot(i)" type="danger"><el-icon :size="14"><Delete /></el-icon></el-button>
                </div>
              </div>
            </div>
          </el-scrollbar>
        </div>
      </div>
    </div>

    <!-- 右侧面板：已累计检测的标签 -->
    <div class="side-panel">
      <ResultPanel
        :detections="accumulatedDetections"
        :inference-time="inferenceTime"
        empty-text="启动摄像头以进行检测"
      />
    </div>

    <!-- 快照预览弹窗：左图右标签 -->
    <el-dialog v-model="previewOpen" title="快照详情" width="85%" top="8vh" :destroy-on-close="true" class="snapshot-dialog">
      <div class="preview-layout" v-if="previewIndex !== null && snapshots[previewIndex]">
        <div class="preview-image-side">
          <img :src="snapshots[previewIndex].dataUrl" class="preview-img-large" />
        </div>
        <div class="preview-info-side">
          <h4>检测标签</h4>
          <div class="preview-tags">
            <el-tag v-for="(l, j) in snapshots[previewIndex].labels" :key="j" size="large" style="margin:2px">{{ l }}</el-tag>
          </div>
          <el-empty v-if="!snapshots[previewIndex].labels?.length" description="无标签" :image-size="40" />
        </div>
      </div>
      <template #footer>
        <el-button @click="previewOpen = false">关闭</el-button>
        <el-button type="primary" @click="downloadSnapshot(previewIndex)">下载</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, onActivated, onDeactivated } from 'vue'
import { Camera, Delete, Download, ZoomIn, Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import ResultPanel from '../components/ResultPanel.vue'

const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000'

const videoRef = ref(null)
const canvasRef = ref(null)
const snapScrollbar = ref(null)
const cameraActive = ref(false)
const wsConnected = ref(false)
const fps = ref(0)
const frameDetections = ref(0)
const inferenceTime = ref(0)
const detections = ref([])
const snapshots = ref([])
const previewOpen = ref(false)
const previewIndex = ref(null)
const devices = ref([])
const selectedDevice = ref('')

// 持久化标签（计数 + 排序）
const labelCountMap = reactive(new Map())  // class_name -> count
const accumulatedDetections = ref([])

function updateAccumulated(dets) {
  for (const d of (dets || [])) {
    const key = d.class_name
    labelCountMap.set(key, (labelCountMap.get(key) || 0) + 1)
  }
  // 按出现次数降序排列，保留第一个检测的 class_id 用于颜色
  const seen = new Map()
  for (const d of (dets || [])) {
    if (!seen.has(d.class_name)) seen.set(d.class_name, d.class_id)
  }
  const sorted = [...labelCountMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({
      class_id: seen.get(name) || 0,
      class_name: name,
      name_cn: dets?.find(d => d.class_name === name)?.name_cn || name,
      count,
    }))
  accumulatedDetections.value = sorted
}

let mediaStream = null
let ws = null
let sendTimer = null
let animationId = null
let videoReady = false
let lastDets = []

const colorMap = [
  '#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6',
  '#1ABC9C','#E67E22','#2980B9','#27AE60','#D35400',
  '#8E44AD','#16A085','#C0392B','#7F8C8D','#F1C40F',
]

function onVideoReady() { videoReady = true }

async function enumerateDevices() {
  try {
    const all = await navigator.mediaDevices.enumerateDevices()
    devices.value = all.filter(d => d.kind === 'videoinput')
    // 自动选择：无选中且有设备 → 选第一个；已选设备已断开 → 切换到第一个或清空
    if (devices.value.length > 0) {
      if (!selectedDevice.value || !devices.value.some(d => d.deviceId === selectedDevice.value)) {
        selectedDevice.value = devices.value[0].deviceId
      }
    } else {
      selectedDevice.value = ''
    }
  } catch (e) {}
}

async function toggleCamera() { cameraActive.value ? stopCamera() : await startCamera() }

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    ElMessage.error('摄像头需要 HTTPS 或 localhost 环境，请使用 http://localhost:8000/ 访问')
    return
  }
  try {
    const constraints = {
      video: { width: { ideal: 640 }, height: { ideal: 480 }, deviceId: selectedDevice.value ? { exact: selectedDevice.value } : undefined },
      audio: false,
    }
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
    videoRef.value.srcObject = mediaStream
    await videoRef.value.play()
    cameraActive.value = true
    videoReady = true
    connectWebSocket()
    drawLoop()
    startSending()
    await enumerateDevices()
  } catch (err) {
    ElMessage.error('无法访问摄像头: ' + err.message)
  }
}

async function switchDevice() {
  if (cameraActive.value) { stopCamera(); await startCamera() }
}

function stopCamera() {
  cameraActive.value = false; wsConnected.value = false; videoReady = false
  clearInterval(sendTimer); cancelAnimationFrame(animationId)
  if (ws) { ws.close(); ws = null }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null }
  const ctx = canvasRef.value?.getContext('2d')
  if (ctx) ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)
  detections.value = []; frameDetections.value = 0; lastDets = []
}

function connectWebSocket() {
  ws = new WebSocket(`${WS_BASE}/api/detect/stream`)
  ws.onopen = () => { wsConnected.value = true }
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'detection') {
        frameDetections.value = data.total; fps.value = data.fps; inferenceTime.value = data.inference_ms
        detections.value = data.detections || []; lastDets = data.detections || []
        drawBoxes(lastDets)
        updateAccumulated(data.detections)
      }
    } catch(e) {}
  }
  ws.onclose = () => { wsConnected.value = false; lastDets = [] }
  ws.onerror = () => { wsConnected.value = false }
}

function startSending() {
  sendTimer = setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN && cameraActive.value) {
      const canvas = document.createElement('canvas'); const video = videoRef.value
      if (!video || video.readyState < 2) return
      canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480
      canvas.getContext('2d').drawImage(video, 0, 0)
      canvas.toBlob(blob => { if (blob) blob.arrayBuffer().then(buf => { if (ws?.readyState === WebSocket.OPEN) ws.send(buf) }) }, 'image/jpeg', 0.65)
    }
  }, 600)
}

function drawLoop() { if (!cameraActive.value) return; drawBoxes(lastDets); animationId = requestAnimationFrame(drawLoop) }

function drawBoxes(dets) {
  const canvas = canvasRef.value; const video = videoRef.value
  if (!canvas || !video || !videoReady) return
  const ctx = canvas.getContext('2d')
  const vw = video.videoWidth || canvas.width; const vh = video.videoHeight || canvas.height
  if (!vw || !vh) return
  canvas.width = vw; canvas.height = vh; ctx.clearRect(0, 0, vw, vh)
  if (!dets || !dets.length) return
  const lnW = Math.max(2, vw / 400)
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.bbox; const color = colorMap[d.class_id % colorMap.length]
    ctx.strokeStyle = color; ctx.lineWidth = lnW; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    const label = `${d.name_cn || d.class_name} ${(d.confidence*100).toFixed(0)}%`
    ctx.font = `bold ${Math.max(13, vw/50)}px sans-serif`; const tw = ctx.measureText(label).width + 6; const th = Math.max(18, vw/40)
    const ly = Math.max(0, y1 - th); ctx.fillStyle = color; ctx.fillRect(x1, ly, tw, th)
    ctx.fillStyle = '#fff'; ctx.fillText(label, x1 + 3, ly + th - 4)
  }
}

function captureSnapshot() {
  const video = videoRef.value; const canvas = canvasRef.value
  if (!video || !canvas) return
  const sc = document.createElement('canvas'); sc.width = video.videoWidth || 640; sc.height = video.videoHeight || 480
  sc.getContext('2d').drawImage(video, 0, 0); sc.getContext('2d').drawImage(canvas, 0, 0)
  const dataUrl = sc.toDataURL('image/jpeg', 0.9)
  // 获取当前标签
  const labels = lastDets.filter(Boolean).map(d => d.name_cn || d.class_name)
  snapshots.value.push({ dataUrl, labels })
  ElMessage.success('快照已保存')
}

function previewSnapshot(i) { previewIndex.value = i; previewOpen.value = true }
function downloadSnapshot(i) {
  if (i === null || !snapshots.value[i]) return
  const a = document.createElement('a'); a.href = snapshots.value[i].dataUrl
  a.download = `snapshot_${Date.now()}.jpg`; a.click()
}
function removeSnapshot(i) { snapshots.value.splice(i, 1) }
function resetAccumulated() { labelCountMap.clear(); accumulatedDetections.value = []; ElMessage.success('目录已刷新') }
function refreshCamera() {
  enumerateDevices()  // 异步后台刷新，不阻塞 UI
  ElMessage.success('摄像头列表已刷新')
}
function onSnapWheel(e) {
  if (!snapScrollbar.value) return
  const wrap = snapScrollbar.value.wrapRef
  if (wrap) {
    snapScrollbar.value.setScrollLeft(wrap.scrollLeft + (e.deltaY || e.deltaX))
  }
}

onMounted(() => { enumerateDevices() })
onUnmounted(() => { stopCamera() })
onDeactivated(() => { if (cameraActive.value) stopCamera() })
</script>

<style scoped>
.detect-view { display: flex; height: 100%; }
.main-area { flex: 1; display: flex; flex-direction: column; padding: 16px; overflow: auto; }
.side-panel { width: 280px; border-left: 1px solid #e4e7ed; background: #fff; overflow-y: auto; }

.camera-container { max-width: 700px; margin: 0 auto; width: 100%; }
.camera-box { position: relative; width: 100%; aspect-ratio: 4/3; background: #000; border-radius: 8px; overflow: hidden; }
.camera-feed { position: absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; }
.overlay-canvas { position: absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
.camera-placeholder { position: absolute; top:0; left:0; width:100%; height:100%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #909399; background: #1a1a2e; }
.camera-info { display: flex; gap: 20px; justify-content: center; padding: 10px 0; font-size: 13px; color: #606266; flex-wrap: wrap; }
.camera-info strong { color: #303133; }
.status-dot { font-weight: 500; font-size: 12px; }
.status-dot.online::before { content:'● '; color:#67c23a; }
.status-dot.offline::before { content:'● '; color:#f56c6c; }
.camera-actions { display: flex; gap: 12px; justify-content: center; align-items: center; padding: 8px 0; flex-wrap: wrap; }

.snapshots-section { margin-top: 12px; }
.section-title { font-size: 13px; font-weight: 600; color: #303133; margin-bottom: 8px; }
.snapshots-row { display: flex; gap: 8px; padding-bottom: 8px; }
/* el-scrollbar 水平滚动条定制 */
.snap-scrollbar { height: auto; }
.snap-scrollbar :deep(.el-scrollbar__wrap) { overflow: hidden; }
.snap-scrollbar :deep(.el-scrollbar__bar.is-horizontal) { height: 6px; }
.snap-scrollbar :deep(.el-scrollbar__bar.is-horizontal .el-scrollbar__thumb) {
  background: #c0c4cc; border-radius: 3px;
}
.snap-scrollbar :deep(.el-scrollbar__bar.is-horizontal .el-scrollbar__thumb:hover) {
  background: #909399;
}
.snapshot-item { position: relative; flex-shrink: 0; border-radius: 6px; overflow: hidden; border: 2px solid #ebeef5; cursor: pointer; width: 140px; }
.snapshot-item:hover { border-color: #409eff; }
.snapshot-img { width: 140px; height: 100px; object-fit: cover; display: block; }
.snapshot-labels { padding: 2px 4px; display: flex; flex-wrap: wrap; gap: 2px; }
.snapshot-btns { display: flex; gap: 4px; justify-content: center; padding: 4px; }
.snapshot-btns .el-button { width: 26px; height: 26px; padding: 0; }

/* 预览弹窗：左图右信息 */
.snapshot-dialog :deep(.el-dialog__body) { padding: 12px; }
.preview-layout { display: flex; gap: 12px; height: 70vh; }
.preview-image-side { flex: 1; min-width: 0; display: flex; align-items: center; justify-content: center; }
.preview-img-large { max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: 6px; }
.preview-info-side { width: 200px; flex-shrink: 0; overflow-y: auto; }
.preview-info-side h4 { margin: 0 0 12px; font-size: 15px; color: #303133; }
.preview-tags { display: flex; flex-wrap: wrap; gap: 4px; }
</style>
