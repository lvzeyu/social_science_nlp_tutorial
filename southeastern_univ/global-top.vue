<template>
  <div v-if="$slidev.nav.currentPage > 1" class="qrcode-fixed">
    <QRCode value="https://sli.dev" :size="40" render-as="svg" />
  </div>
</template>

<style scoped>
.qrcode-fixed {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 100;
}
</style>
