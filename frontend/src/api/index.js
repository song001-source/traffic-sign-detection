/**
 * API 服务层
 * 封装所有后端接口调用
 */
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,  // 大文件上传需较长超时
})

/**
 * 图片检测
 * @param {File} file - 图片文件
 * @param {Object} params - { conf, iou, imgsz }
 * @returns {Promise}
 */
export function detectImage(file, params = {}) {
  const formData = new FormData()
  formData.append('file', file)
  const query = new URLSearchParams()
  if (params.conf) query.set('conf', params.conf)
  if (params.iou) query.set('iou', params.iou)
  if (params.imgsz) query.set('imgsz', params.imgsz)
  return api.post(`/api/detect/image?${query.toString()}`, formData)
}

/**
 * 视频检测（带实时进度）
 * 1. 提交视频 → 获得 task_id
 * 2. 轮询 getVideoProgress(task_id) 获取进度
 * 3. 完成后从 result_url 获取结果视频
 */
export function submitVideo(file, params = {}) {
  const formData = new FormData()
  formData.append('file', file)
  const query = new URLSearchParams()
  if (params.conf) query.set('conf', params.conf)
  if (params.iou) query.set('iou', params.iou)
  return api.post(`/api/detect/video/process?${query.toString()}`, formData)
}

/**
 * 查询视频处理进度
 */
export function getVideoProgress(taskId) {
  return api.get(`/api/detect/video/progress/${taskId}`)
}
export function getTaskStatus(taskId) {
  return api.get(`/api/detect/task/${taskId}`)
}

/**
 * 构建标注图完整URL
 */
export function getAnnotatedUrl(path) {
  if (!path) return ''
  if (path.startsWith('http')) return path
  return `${API_BASE}${path}`
}

/**
 * 获取模型列表
 */
export function getModels() {
  return api.get('/api/models')
}

/**
 * 健康检查
 */
export function healthCheck() {
  return api.get('/api/health')
}

export default api
