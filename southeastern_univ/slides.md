---
# try also 'default' to start simple
theme: neversink
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
#background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: 自然语言处理在社会科学中的应用
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# duration of the presentation
duration: 20min
color: navy-light
layout: intro
colorSchema: light

fonts:
  sans: 'Noto Sans SC'
  serif: 'Noto Serif SC'
  mono: 'Roboto Mono'
  provider: google

css: unocss

addons:
  - "@katzumi/slidev-addon-qrcode"
---

<style src="./style.css"></style>

# 自然语言处理在社会科学中的应用

## 从词向量到大语言模型

### 吕 泽宇 / Zeyu Lyu <a href="https://researchmap.jp/lyuzeyu?lang=en" class="ns-c-iconlink"><mdi-open-in-new /></a>  

2025年12月8日・东南大学

<div style="margin-top: 3rem;">
<QRCode value="https://sli.dev" :size="100" render-as="svg" />
</div>


---
transition: fade-out
---

# 自我介绍

- **所属**: 日本东北大学 文学研究科 [计算人文社会学研究室](https://www.sal.tohoku.ac.jp/jp/research/researcher/profile/---id-190.html) 副教授
- **主要经历**: 博士期间隶属于[日本学术振兴会特别研究员DC2](https://www.jsps.go.jp/j-pd/), [東北大学WISE Program for AI Electronics](https://www.aie.tohoku.ac.jp/), [東北大学Division for Interdisciplinary Advanced Research and Education](http://www.iiare.tohoku.ac.jp/); 在担任[东京大学社会科学研究所](https://jww.iss.u-tokyo.ac.jp/)研究员之后加入目前所属单位.

- **主要研究方向**
    - 基于大规模移动数据关于社会空间隔离(Social spatial segregation)的实证分析和社会模拟
    - <v-click><span class="normal highlight">关于网络空间上意见形成的实证分析和社会模拟</span></v-click>
    - <v-after><span class="normal highlight">基于文本的文化演化分析</span></v-after>

<p v-click class="absolute bottom-45 left-150 transform" style="color: #146b8c;">
  自然语言处理在社会科学中的应用
</p>

<arrow
    v-after
    x1="480"
    y1="345"
    x2="580"
    y2="345"
    color="#146b8c"
    width="3"
    arrowSize="1" />

<style>
.normal {
  transition: color 0.5s ease-in-out;
}
.highlight {
  color: black !important;
  font-weight: bold;
  text-decoration: underline;
}
</style>

<!--

-->