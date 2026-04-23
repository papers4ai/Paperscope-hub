# AI Paper Hub

> 自动采集、整理世界模型、物理机理、医疗AI领域的最新论文

[English](README.md) | [中文](README_zh.md)

## 特性

- **自动采集** - 从 arXiv 抓取论文，支持增量更新
- **智能分类** - 基于关键词规则分类到三个领域：
  - 世界模型 (World Model)
  - 物理机理 (Physical AI)
  - 医疗AI (Medical AI)
- **任务标注** - 自动标注论文任务类型
- **免费托管** - 部署在 GitHub Pages
- **可选升级** - 支持本地语义搜索（免费）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 采集论文
python main.py

# 处理和分类
python pipeline.py

# 预览
python3 -m http.server 8080
```

打开 http://localhost:8080 查看

## 项目结构

```
paper-hub/
├── main.py                 # 论文采集
├── pipeline.py             # 处理和分类
├── config.py               # 配置和关键词
├── semantic_search.py      # 语义搜索（可选）
├── index.html              # 前端界面
├── .github/workflows/      # 自动更新
└── output/                 # 网站运行与 Pages 部署使用的数据输出目录
    ├── papers.json         # 全部论文
    ├── papers_world_model.json
    ├── papers_physical_ai.json
    └── papers_medical_ai.json

说明：前端页面仅加载 `output/` 下的数据文件（如 `output/papers.json`、`output/task_meta.json`、`output/trending.json`）。`outputs/` 为其他产物目录，不参与网站数据加载。
```

## 部署到 GitHub Pages

1. 创建 GitHub 仓库
2. 推送代码
3. Settings → Pages → Source: GitHub Actions
4. 等待 Actions 自动运行

## 可选升级：语义搜索

```bash
# 安装语义搜索依赖
pip install sentence-transformers numpy

# 测试语义搜索
python semantic_search.py "World model for robot manipulation"
```

## 成本

| 功能 | 成本 |
|------|------|
| 基础版 | 免费 |
| 语义搜索 | 免费（本地模型）|
| GitHub托管 | 免费 |
| GitHub Actions | 免费（2000分钟/月）|

## License

MIT