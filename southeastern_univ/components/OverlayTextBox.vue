<template>
  <div 
    class="overlay-text-box"
    :style="{
      position: position,
      top: top,
      left: left,
      right: right,
      bottom: bottom,
      width: width,
      maxWidth: maxWidth,
      padding: padding,
      backgroundColor: bgColor,
      borderColor: borderColor,
      borderRadius: borderRadius,
      zIndex: zIndex
    }"
  >
    <slot />
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  // 位置属性
  position: {
    type: String,
    default: 'absolute'
  },
  top: {
    type: String,
    default: 'auto'
  },
  left: {
    type: String,
    default: 'auto'
  },
  right: {
    type: String,
    default: 'auto'
  },
  bottom: {
    type: String,
    default: 'auto'
  },
  // 尺寸属性
  width: {
    type: String,
    default: 'auto'
  },
  maxWidth: {
    type: String,
    default: '600px'
  },
  padding: {
    type: String,
    default: '1rem 1.5rem'
  },
  // 颜色主题 - 对应 slidev 主题颜色
  theme: {
    type: String,
    default: 'indigo-light', // indigo-light, navy-light, amber, teal 等
    validator: (value) => {
      return ['indigo-light', 'navy-light', 'amber', 'teal', 'rose'].includes(value)
    }
  },
  // 透明度
  opacity: {
    type: Number,
    default: 0.95
  },
  // 边框圆角
  borderRadius: {
    type: String,
    default: '12px'
  },
  // z-index
  zIndex: {
    type: Number,
    default: 10
  }
})

// 根据主题设置背景色和边框色
const bgColor = computed(() => {
  const themes = {
    'indigo-light': `rgba(238, 242, 255, ${props.opacity})`, // 淡紫蓝色
    'navy-light': `rgba(237, 242, 247, ${props.opacity})`,   // 淡海军蓝
    'amber': `rgba(254, 243, 199, ${props.opacity})`,        // 琥珀色
    'teal': `rgba(204, 251, 241, ${props.opacity})`,         // 青色
    'rose': `rgba(255, 241, 242, ${props.opacity})`          // 玫瑰色
  }
  return themes[props.theme] || themes['indigo-light']
})

const borderColor = computed(() => {
  const themes = {
    'indigo-light': 'rgba(99, 102, 241, 0.3)',  // indigo-500
    'navy-light': 'rgba(59, 130, 246, 0.3)',    // blue-500
    'amber': 'rgba(245, 158, 11, 0.3)',         // amber-500
    'teal': 'rgba(20, 184, 166, 0.3)',          // teal-500
    'rose': 'rgba(244, 63, 94, 0.3)'            // rose-500
  }
  return themes[props.theme] || themes['indigo-light']
})
</script>

<style scoped>
.overlay-text-box {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  border: 2px solid;
  backdrop-filter: blur(8px);
  transition: all 0.3s ease;
}

.overlay-text-box:hover {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
  transform: translateY(-2px);
}

/* 文本样式 */
.overlay-text-box h1,
.overlay-text-box h2,
.overlay-text-box h3,
.overlay-text-box h4 {
  margin: 0;
  font-family: 'Noto Sans SC', sans-serif;
}

.overlay-text-box p {
  margin: 0.5rem 0;
  font-family: 'Noto Sans SC', sans-serif;
}

/* 不同主题的标题颜色 */
.overlay-text-box h1,
.overlay-text-box h2,
.overlay-text-box h3 {
  color: #4338ca; /* indigo-700 作为默认 */
  font-weight: 600;
}
</style>
