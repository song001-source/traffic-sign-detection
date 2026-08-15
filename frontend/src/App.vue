<template>
  <div class="app-container">
    <!-- 顶部导航 -->
    <header class="app-header">
      <div class="logo">
        <el-icon :size="22"><Aim /></el-icon>
        <span class="logo-text">交通标志检测系统</span>
      </div>
      <div class="header-tabs">
        <button
          v-for="tab in tabs"
          :key="tab.path"
          class="tab-btn"
          :class="{ active: currentTab === tab.path }"
          @click="switchTab(tab.path)"
        >
          <el-icon :size="16"><component :is="tab.icon" /></el-icon>
          <span>{{ tab.label }}</span>
        </button>
      </div>
      <div class="header-spacer" />
    </header>

    <!-- 内容区（KeepAlive 保持状态） -->
    <main class="app-main">
      <router-view v-slot="{ Component }">
        <keep-alive>
          <component :is="Component" />
        </keep-alive>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Aim } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()

const tabs = [
  { path: '/detect/image', label: '图片检测', icon: 'PictureFilled' },
  { path: '/detect/video', label: '视频检测', icon: 'VideoCameraFilled' },
  { path: '/detect/camera', label: '实时摄像头', icon: 'Monitor' },
]

const currentTab = computed(() => route.path)

function switchTab(path) {
  router.push(path)
}
</script>

<style scoped>
.app-container { display: flex; flex-direction: column; height: 100vh; background: #f5f7fa; }
.app-header {
  display: flex; align-items: center; height: 52px; padding: 0 20px;
  background: #fff; border-bottom: 1px solid #e4e7ed; flex-shrink: 0; gap: 24px;
}
.logo { display: flex; align-items: center; gap: 8px; color: #303133; }
.logo-text { font-size: 15px; font-weight: 600; }
.header-tabs { display: flex; gap: 4px; }
.tab-btn {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 16px; border: none; background: transparent;
  border-radius: 6px; font-size: 13px; color: #606266; cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.tab-btn:hover { background: #f0f2f5; color: #303133; }
.tab-btn.active { background: #ecf5ff; color: #409eff; font-weight: 500; }
.header-spacer { flex: 1; }
.app-main { flex: 1; overflow: hidden; }
</style>
