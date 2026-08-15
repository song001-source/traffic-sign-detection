<template>
  <div class="result-panel">
    <!-- 结果概览 -->
    <div v-if="detections.length > 0" class="result-header">
      <div class="result-count">
        <strong>{{ detections.length }}</strong> {{ isCountMode ? '种标志' : '个标志' }}
      </div>
      <div v-if="inferenceTime" class="result-time">
        {{ inferenceTime }}ms
      </div>
    </div>

    <!-- 检测结果列表 -->
    <div v-if="detections.length > 0" class="result-list">
      <div
        v-for="(d, idx) in detections"
        :key="idx"
        class="result-item"
        :style="{ borderLeftColor: colorMap[d.class_id % colorMap.length] }"
      >
        <span
          class="result-dot"
          :style="{ background: colorMap[d.class_id % colorMap.length] }"
        />
        <div class="result-text">
          <span class="result-name">{{ d.name_cn || d.class_name }}</span>
          <span class="result-code">{{ d.class_name }}</span>
        </div>
        <span class="result-conf" :class="confidenceClass(d.confidence)">
          <template v-if="d.count !== undefined">×{{ d.count }}</template>
          <template v-else-if="d.confidence !== undefined">{{ (d.confidence * 100).toFixed(0) }}%</template>
        </span>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <el-empty :image-size="60" :description="emptyText" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  detections: { type: Array, default: () => [] },
  inferenceTime: { type: Number, default: 0 },
  emptyText: { type: String, default: '上传图片开始检测' },
})

const isCountMode = computed(() => props.detections.length > 0 && props.detections[0]?.count !== undefined)

const colorMap = [
  '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
  '#1ABC9C', '#E67E22', '#2980B9', '#27AE60', '#D35400',
  '#8E44AD', '#16A085', '#C0392B', '#7F8C8D', '#F1C40F',
]

function confidenceClass(conf) {
  if (conf >= 0.8) return 'conf-high'
  if (conf >= 0.5) return 'conf-mid'
  return 'conf-low'
}
</script>

<style scoped>
.result-panel { height: 100%; display: flex; flex-direction: column; }
.result-header {
  padding: 12px 16px;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid #ebeef5;
}
.result-count { font-size: 14px; color: #303133; }
.result-count strong { font-size: 20px; color: #409EFF; margin-right: 4px; }
.result-time { font-size: 12px; color: #67C23A; font-family: monospace; }
.result-list { flex: 1; overflow-y: auto; }
.result-item {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; margin: 2px 8px;
  border-radius: 4px; border-left: 3px solid;
  background: #fafafa; transition: background 0.15s;
}
.result-item:hover { background: #f0f2f5; }
.result-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.result-text { flex: 1; min-width: 0; }
.result-name { display: block; font-size: 13px; font-weight: 500; color: #303133; }
.result-code { display: block; font-size: 11px; color: #909399; font-family: monospace; }
.result-conf { font-size: 13px; font-weight: 600; font-family: monospace; }
.conf-high { color: #E74C3C; }
.conf-mid { color: #F39C12; }
.conf-low { color: #909399; }
.empty-state {
  flex: 1; display: flex; align-items: center; justify-content: center;
}
</style>
