<template>
  <div class="detect-view">
    <!-- 主区域 -->
    <div class="main-area">
      <!-- 未上传 -->
      <div v-if="!imageLoaded" class="upload-zone" @dragover.prevent @drop.prevent="handleDrop">
        <el-icon class="upload-icon" :size="56"><UploadFilled /></el-icon>
        <p class="upload-title">拖拽图片到此处</p>
        <p class="upload-sub">或点击下方按钮选择文件</p>
        <el-button type="primary" size="large" @click="triggerUpload">
          选择图片
        </el-button>
        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          style="display:none"
          @change="onFileInput"
        />
        <p class="upload-hint">支持 JPG / PNG / WebP，最大 20MB</p>
      </div>

      <!-- 结果展示 -->
      <div v-else class="result-area">
        <!-- 原图 vs 检测结果 -->
        <div class="compare-row">
          <div class="compare-panel">
            <div class="compare-label">原图</div>
            <img :src="originalUrl" class="compare-image" />
          </div>
          <div class="compare-panel">
            <div class="compare-label">检测结果</div>
            <img v-if="annotatedUrl" :src="annotatedUrl" class="compare-image" />
            <div v-else-if="loading" class="compare-placeholder">
              <el-icon class="is-loading" :size="32"><Loading /></el-icon>
              <p>检测中...</p>
            </div>
            <div v-else class="compare-placeholder">
              <p>点击"开始检测"</p>
            </div>
          </div>
        </div>

        <!-- 操作栏 -->
        <div class="action-bar">
          <el-button size="large" plain @click="resetImage" class="btn-reset">重新选择</el-button>
          <el-button type="primary" size="large" :loading="loading" @click="doDetect">
            {{ loading ? '检测中...' : '开始检测' }}
          </el-button>
        </div>
      </div>
    </div>

    <!-- 右侧结果面板 -->
    <div class="side-panel">
      <ResultPanel :detections="detections" :inference-time="inferenceTime" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled, Loading } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { detectImage, getAnnotatedUrl } from '../api'
import ResultPanel from '../components/ResultPanel.vue'

const fileInput = ref(null)
const originalUrl = ref('')
const annotatedUrl = ref('')
const detections = ref([])
const inferenceTime = ref(0)
const loading = ref(false)
const imageLoaded = ref(false)
const currentFile = ref(null)

function triggerUpload() { fileInput.value?.click() }
function onFileInput(e) { if (e.target.files[0]) loadFile(e.target.files[0]) }
function handleDrop(e) { const f = e.dataTransfer.files[0]; if (f) loadFile(f) }

function loadFile(file) {
  currentFile.value = file
  originalUrl.value = URL.createObjectURL(file)
  annotatedUrl.value = ''
  detections.value = []
  inferenceTime.value = 0
  imageLoaded.value = true
}

function resetImage() {
  currentFile.value = null
  originalUrl.value = ''
  annotatedUrl.value = ''
  detections.value = []
  inferenceTime.value = 0
  imageLoaded.value = false
}

async function doDetect() {
  if (!currentFile.value) return
  loading.value = true
  try {
    const res = await detectImage(currentFile.value, { conf: 0.35, iou: 0.45 })
    const data = res.data
    detections.value = data.detections || []
    inferenceTime.value = data.inference_time_ms
    annotatedUrl.value = getAnnotatedUrl(data.annotated_image_url)
  } catch (err) {
    ElMessage.error('检测失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.detect-view { display: flex; height: 100%; }
.main-area { flex: 1; display: flex; flex-direction: column; padding: 24px 16px; overflow: auto; }
.side-panel { width: 280px; border-left: 1px solid #e4e7ed; background: #fff; overflow-y: auto; }

/* Upload zone */
.upload-zone {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  margin: 0 80px;
  border: 2px dashed #dcdfe6; border-radius: 12px;
  transition: border-color 0.3s;
  cursor: pointer;
}
.upload-zone:hover { border-color: #409eff; background: #f0f7ff; }
.upload-icon { color: #c0c4cc; margin-bottom: 16px; }
.upload-title { font-size: 18px; color: #303133; margin: 0 0 4px; }
.upload-sub { font-size: 13px; color: #909399; margin: 0 0 16px; }
.upload-hint { font-size: 12px; color: #c0c4cc; margin-top: 16px; }

/* Result area */
.result-area { flex: 1; display: flex; flex-direction: column; }
.compare-row { flex: 1; display: flex; gap: 16px; min-height: 0; }
.compare-panel {
  flex: 1; display: flex; flex-direction: column;
  background: #f5f7fa; border-radius: 8px; overflow: hidden;
}
.compare-label {
  padding: 8px 12px; font-size: 13px; font-weight: 600;
  color: #303133; background: #ebeef5;
}
.compare-image {
  flex: 1; width: 100%; object-fit: contain; background: #000;
}
.compare-placeholder {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center; color: #909399;
}
.compare-placeholder p { margin-top: 8px; }
.action-bar {
  display: flex; gap: 16px; justify-content: center; padding: 16px 0; align-items: center;
}
.btn-reset { font-size: 15px; padding: 10px 28px; border-width: 1.5px; height: 42px; }
</style>
