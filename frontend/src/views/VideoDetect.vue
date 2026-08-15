<template>
  <div class="detect-view">
    <div class="main-area">
      <div v-if="!videoSelected" class="upload-zone" @dragover.prevent @drop.prevent="handleDrop">
        <el-icon class="upload-icon" :size="56"><VideoCamera /></el-icon>
        <p class="upload-title">拖拽视频到此处</p>
        <p class="upload-sub">或点击下方按钮选择文件</p>
        <el-button type="primary" size="large" @click="triggerUpload">选择视频</el-button>
        <input ref="fileInput" type="file" accept="video/*" style="display:none" @change="onFileInput" />
        <p class="upload-hint">支持 MP4 / AVI / MOV，建议 &lt; 100MB</p>
      </div>

      <div v-else class="result-area">
        <div class="compare-row">
          <div class="compare-panel">
            <div class="compare-label">原视频</div>
            <video :src="videoUrl" controls class="compare-video" />
          </div>
          <div class="compare-panel">
            <div class="compare-label">检测结果</div>
            <video v-if="resultVideoUrl" :src="resultVideoUrl" controls class="compare-video" />
            <div v-else-if="taskStatus === 'pending' || taskStatus === 'processing'" class="compare-placeholder">
              <el-progress :percentage="realProgress" :stroke-width="10" :status="taskStatus === 'failed' ? 'exception' : ''" style="width:220px" />
              <p class="progress-label">处理中，请稍候...</p>
            </div>
            <div v-else class="compare-placeholder"><p>点击"开始检测"</p></div>
          </div>
        </div>

        <!-- 关键帧 -->
        <div v-if="keyFrames.length > 0" class="keyframes-section">
          <div class="compare-label">关键帧 ({{ keyFrames.length }})</div>
          <el-scrollbar ref="kfScrollbar" class="kf-scrollbar" @wheel.prevent="onKfWheel">
            <div class="keyframes-row">
              <div v-for="(kf, i) in keyFrames" :key="i" class="keyframe-item" @click="previewKeyFrame(i)">
                <img :src="kf.url" class="keyframe-img" />
                <div class="keyframe-labels">
                  <el-tag size="small" type="primary">{{ kf.label }} ({{ (kf.confidence*100).toFixed(0) }}%)</el-tag>
                </div>
                <div class="keyframe-btns">
                  <el-button size="small" circle @click.stop="previewKeyFrame(i)"><el-icon :size="14"><ZoomIn /></el-icon></el-button>
                  <el-button size="small" circle @click.stop="downloadKeyFrame(i)"><el-icon :size="14"><Download /></el-icon></el-button>
                  <el-button size="small" circle @click.stop="removeKeyFrame(i)" type="danger"><el-icon :size="14"><Delete /></el-icon></el-button>
                </div>
              </div>
            </div>
          </el-scrollbar>
        </div>

        <div class="action-bar">
          <el-button size="large" plain @click="resetVideo" class="btn-reset">重新选择</el-button>
          <el-button type="primary" size="large" :loading="taskStatus === 'pending' || taskStatus === 'processing'" @click="doDetect" :disabled="taskStatus === 'processing'">
            开始检测
          </el-button>
        </div>
      </div>
    </div>

    <!-- 关键帧预览弹窗：左图右标签 -->
    <el-dialog v-model="previewOpen" title="关键帧详情" width="85%" top="8vh" :destroy-on-close="true" class="kf-dialog">
      <div class="preview-layout" v-if="previewIndex !== null">
        <div class="preview-image-side">
          <img :src="keyFrames[previewIndex]?.url" class="preview-img-large" />
        </div>
        <div class="preview-info-side">
          <h4>检测标签</h4>
          <div v-if="keyFrames[previewIndex]" class="preview-tags">
            <el-tag type="primary" size="large" style="margin:4px">
              {{ keyFrames[previewIndex].label }} ({{ (keyFrames[previewIndex].confidence*100).toFixed(0) }}%)
            </el-tag>
          </div>
          <el-divider />
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="帧号">{{ keyFrames[previewIndex]?.frame_idx || '-' }}</el-descriptions-item>
            <el-descriptions-item label="类别">{{ keyFrames[previewIndex]?.label || '-' }}</el-descriptions-item>
            <el-descriptions-item label="置信度">{{ keyFrames[previewIndex] ? (keyFrames[previewIndex].confidence*100).toFixed(1)+'%' : '-' }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
      <template #footer>
        <el-button @click="previewOpen = false">关闭</el-button>
        <el-button type="primary" @click="downloadKeyFrame(previewIndex)">下载</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue'
import { VideoCamera, ZoomIn, Download, Delete } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { submitVideo, getVideoProgress, getAnnotatedUrl } from '../api'

const fileInput = ref(null)
const videoUrl = ref('')
const videoSelected = ref(false)
const resultVideoUrl = ref('')
const taskStatus = ref('')
const realProgress = ref(0)
const taskError = ref('')
const keyFrames = ref([])
const currentFile = ref(null)
const previewOpen = ref(false)
const previewIndex = ref(null)
const kfScrollbar = ref(null)
let pollTimer = null

function triggerUpload() { fileInput.value?.click() }
function onFileInput(e) { if (e.target.files[0]) loadFile(e.target.files[0]) }
function handleDrop(e) { const f = e.dataTransfer.files[0]; if (f) loadFile(f) }

function loadFile(file) {
  currentFile.value = file; videoUrl.value = URL.createObjectURL(file)
  videoSelected.value = true; resultVideoUrl.value = ''; taskStatus.value = ''
  realProgress.value = 0; taskError.value = ''; keyFrames.value = []; clearPoll()
}

function resetVideo() {
  videoUrl.value = ''; videoSelected.value = false; resultVideoUrl.value = ''
  taskStatus.value = ''; realProgress.value = 0; keyFrames.value = []; currentFile.value = null; clearPoll()
}

function clearPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null } }

async function doDetect() {
  if (!currentFile.value) return
  clearPoll(); taskStatus.value = 'pending'; realProgress.value = 0
  try {
    const res = await submitVideo(currentFile.value, { conf: 0.35, iou: 0.45 })
    const taskId = res.data.task_id
    pollTimer = setInterval(async () => {
      try {
        const pr = await getVideoProgress(taskId)
        const d = pr.data
        taskStatus.value = d.status; realProgress.value = d.progress
        if (d.status === 'completed') {
          clearPoll()
          if (d.result) {
            resultVideoUrl.value = getAnnotatedUrl(d.result.result_url)
            if (d.result.key_frames?.length) {
              keyFrames.value = d.result.key_frames.map(kf => ({
                ...kf,
                url: getAnnotatedUrl(kf.url),
                label: kf.name_cn || kf.class_name,
              }))
            }
            ElMessage.success(`完成：${d.result.total_frames} 帧，${d.result.total_detections} 目标，${keyFrames.value.length} 关键帧`)
          }
        } else if (d.status === 'failed') {
          clearPoll(); taskError.value = d.error || '处理失败'; ElMessage.error(taskError.value)
        }
      } catch (e) {
        if (e.response?.status === 404) {
          clearPoll()
          taskStatus.value = 'failed'
          taskError.value = '任务已过期（可能服务器重启），请重新提交'
          ElMessage.warning(taskError.value)
        }
      }
    }, 500)
  } catch (err) {
    taskStatus.value = 'failed'; taskError.value = err.response?.data?.detail || err.message
    ElMessage.error('提交失败: ' + taskError.value)
  }
}

function previewKeyFrame(i) { previewIndex.value = i; previewOpen.value = true }
function downloadKeyFrame(i) {
  if (i === null || !keyFrames.value[i]) return
  const a = document.createElement('a'); a.href = keyFrames.value[i].url
  a.download = `keyframe_${Date.now()}.jpg`; a.click()
}
function removeKeyFrame(i) { keyFrames.value.splice(i, 1) }
function onKfWheel(e) {
  if (!kfScrollbar.value) return
  const wrap = kfScrollbar.value.wrapRef
  if (wrap) {
    kfScrollbar.value.setScrollLeft(wrap.scrollLeft + (e.deltaY || e.deltaX))
  }
}

onUnmounted(() => { clearPoll() })
</script>

<style scoped>
.detect-view { display: flex; height: 100%; }
.main-area { flex: 1; display: flex; flex-direction: column; padding: 24px 16px; overflow: auto; }
.upload-zone {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  margin: 0 80px; border: 2px dashed #dcdfe6; border-radius: 12px; cursor: pointer;
}
.upload-zone:hover { border-color: #409eff; background: #f0f7ff; }
.upload-icon { color: #c0c4cc; margin-bottom: 16px; }
.upload-title { font-size: 18px; color: #303133; margin: 0 0 4px; }
.upload-sub { font-size: 13px; color: #909399; margin: 0 0 16px; }
.upload-hint { font-size: 12px; color: #c0c4cc; margin-top: 16px; }
.result-area { flex: 1; display: flex; flex-direction: column; }
.compare-row { flex: 1; display: flex; gap: 16px; min-height: 0; }
.compare-panel { flex: 1; display: flex; flex-direction: column; background: #f5f7fa; border-radius: 8px; overflow: hidden; }
.compare-label { padding: 8px 12px; font-size: 13px; font-weight: 600; color: #303133; background: #ebeef5; }
.compare-video { flex: 1; width: 100%; background: #000; }
.compare-placeholder { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #909399; }
.progress-label { margin-top: 10px; color: #606266; font-size: 13px; text-align: center; }
.keyframes-section { margin-top: 12px; }
.keyframes-row { display: flex; gap: 8px; padding: 8px 0; }
/* el-scrollbar 水平滚动条定制 */
.kf-scrollbar { height: auto; }
.kf-scrollbar :deep(.el-scrollbar__wrap) { overflow: hidden; }
.kf-scrollbar :deep(.el-scrollbar__bar.is-horizontal) { height: 6px; }
.kf-scrollbar :deep(.el-scrollbar__bar.is-horizontal .el-scrollbar__thumb) {
  background: #c0c4cc; border-radius: 3px;
}
.kf-scrollbar :deep(.el-scrollbar__bar.is-horizontal .el-scrollbar__thumb:hover) {
  background: #909399;
}
.keyframe-item { flex-shrink: 0; border-radius: 6px; overflow: hidden; border: 2px solid #ebeef5; cursor: pointer; width: 150px; }
.keyframe-item:hover { border-color: #409eff; }
.keyframe-img { width: 150px; height: 110px; object-fit: cover; display: block; }
.keyframe-labels { padding: 2px 4px; }
.keyframe-btns { display: flex; gap: 4px; justify-content: center; padding: 4px; }
.keyframe-btns .el-button { width: 26px; height: 26px; padding: 0; }
.action-bar { display: flex; gap: 16px; justify-content: center; padding: 16px 0; align-items: center; }
.btn-reset { font-size: 15px; padding: 10px 28px; border-width: 1.5px; height: 42px; }

/* 预览弹窗：左图右信息 */
.kf-dialog :deep(.el-dialog__body) { padding: 12px; }
.preview-layout { display: flex; gap: 12px; height: 70vh; }
.preview-image-side { flex: 1; min-width: 0; display: flex; align-items: center; justify-content: center; }
.preview-img-large { max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: 6px; }
.preview-info-side { width: 200px; flex-shrink: 0; overflow-y: auto; }
.preview-info-side h4 { margin: 0 0 12px; font-size: 15px; color: #303133; }
.preview-tags { display: flex; flex-wrap: wrap; gap: 4px; }
</style>
