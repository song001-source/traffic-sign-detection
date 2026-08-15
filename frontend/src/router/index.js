import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/detect/image',
  },
  {
    path: '/detect/image',
    name: 'ImageDetect',
    component: () => import('../views/ImageDetect.vue'),
    meta: { title: '图片检测', icon: 'PictureFilled' },
  },
  {
    path: '/detect/video',
    name: 'VideoDetect',
    component: () => import('../views/VideoDetect.vue'),
    meta: { title: '视频检测', icon: 'VideoCameraFilled' },
  },
  {
    path: '/detect/camera',
    name: 'CameraDetect',
    component: () => import('../views/CameraDetect.vue'),
    meta: { title: '实时摄像头', icon: 'Monitor' },
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router
