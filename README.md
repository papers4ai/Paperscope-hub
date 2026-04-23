# AI Paper Hub

> 自动采集、整理世界模型、物理机理、医疗AI领域的最新论文

[English](README.md) | [中文](README_zh.md)

## 特性

- **多数据源支持** - OpenAlex API（推荐）和 arXiv API
- **智能关键词** - 使用 LLM 自动生成/更新搜索关键词
- **自动分类** - 基于关键词规则分类到三个领域
- **免费托管** - 部署在 GitHub Pages
- **自动更新** - GitHub Actions 每天自动抓取最新论文

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 采集论文（默认使用 OpenAlex）
python main.py

# 处理和分类
python pipeline.py

# 预览
python3 -m http.server 8080
```

打开 http://localhost:8080 查看

---

## 完整操作指南

### 1. 论文抓取 (`main.py`)

#### 数据源选择

| 数据源 | 参数 | 特点 |
|--------|------|------|
| OpenAlex | `--source openalex`（默认） | 限制宽松、速度快、推荐使用 |
| arXiv | `--source arxiv` | 官方 API、限制严格 |

#### 运行模式

```bash
# 增量更新（默认）- 只抓最近 30 天
python main.py

# 更新关键词 + 增量更新
python main.py --update-keywords

# 全量抓取（补全缺失年份）
python main.py --full

# 只抓指定年份
python main.py --year 2025

# 抓取多个年份
python main.py --years 2023,2024,2025

# 使用 arXiv API
python main.py --source arxiv

# arXiv API + 全量
python main.py --source arxiv --full
```

#### 工作流程

```
python main.py --update-keywords
         ↓
┌────────────────────────────┐
│ 1. 更新关键词（可选）        │
│    keyword_generator.py     │
│    使用 GLM 自动生成关键词   │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 2. 加载关键词               │
│    从 keywords.json 读取    │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 3. 搜索论文                 │
│    OpenAlex/arXiv API       │
│    按关键词搜索 arXiv 论文   │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 4. 合并数据                 │
│    与已有论文去重合并        │
│    保存到 progress.json     │
└────────────────────────────┘
```

---

### 2. 关键词管理 (`keyword_generator.py`)

关键词存储在 `keywords.json`，可手动编辑或自动生成。

```bash
# 查看当前关键词
python keyword_generator.py --show

# 使用 LLM 自动更新关键词
python keyword_generator.py --update
```

#### 关键词结构

```json
{
  "world_model": ["world model", "video generation", ...],
  "physical_ai": ["PINN", "neural operator", ...],
  "medical_ai": ["medical imaging AI", ...]
}
```

#### 手动编辑

直接编辑 `keywords.json` 添加/删除关键词。

---

### 3. 数据处理 (`pipeline.py`)

```bash
# 完整运行（包含 GLM 生成 trending）
python pipeline.py

# 跳过 trending（快速）
python pipeline.py --skip-trending

# 指定输入文件
python pipeline.py --input output/papers.json
```

####流程

```
pipeline.py
         ↓
┌────────────────────────────┐
│ 1. 读取论文数据             │
│    progress.json           │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 2. 领域分类                 │
│    world_model             │
│    physical_ai             │
│    medical_ai              │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 3. 任务标注                 │
│    VidGen, NeRF, PINN...   │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 4. 生成 trending（可选）    │
│    使用 GLM 生成热点论文    │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ 5. 输出文件                 │
│    papers.json             │
│    papers_*.json           │
│    statistics.json         │
│    trending.json           │
│    feed.xml                │
└────────────────────────────┘
```

---

### 4. 完整运行流程

```bash
# 日常更新（推荐）
python main.py --update-keywords && python pipeline.py

# 推送到 GitHub
git add output/ keywords.json
git commit -m "chore: update papers"
git push
```

---

## 项目结构

```
paper-hub/
├── main.py                 # 论文采集
├── pipeline.py             # 处理和分类
├── config.py               # 领域定义和任务标签
├── keyword_generator.py    # 关键词生成器
├── keywords.json           # 搜索关键词（可编辑）
├── openalex_scraper.py     # OpenAlex API 抓取器
├── config_search.py        # arXiv API 抓取器
├── index.html              # 前端界面
├── .github/workflows/      # 自动更新
└── output/                 # 输出目录
    ├── progress.json       # 原始论文数据
    ├── papers.json         # 全部论文（处理后）
    ├── papers_world_model.json
    ├── papers_physical_ai.json
    ├── papers_medical_ai.json
    ├── statistics.json     # 统计信息
    ├── trending.json       # 热点论文
    └── feed.xml            # RSS 订阅
```

---

## GitHub Actions 自动化

`.github/workflows/daily-update.yml` 配置：

- **触发条件**：
  - 每天 UTC 02:05 和 10:05（北京时间 10:05 和 18:05）
  - 手动触发
  - Push 到 main 分支（跳过抓取，只更新前端）

- **自动执行**：
  1. 更新关键词（使用 GLM）
  2. 抓取最新论文（OpenAlex增量）
  3. 生成前端数据
  4. 提交并部署到 GitHub Pages

---

## 领域分类

### World Model（世界模型）

| 任务 | 关键词 |
|------|--------|
| World Model Core | world model, world action model, learned simulator |
| Action Model | action model, affordance, action prediction |
| Foundation Model | foundation model, LLM, VLM |
| Video Generation | video generation, video prediction, video diffusion |
| NeRF/3DGS | NeRF, gaussian splatting, 3D reconstruction |
| MBRL | model-based RL, planning, MPC |

### Physical AI（物理人工智能）

| 任务 | 关键词 |
|------|--------|
| Physics-Informed | PINN, physics-informed neural network |
| Neural Operator | FNO, DeepONet, neural operator |
| Robot Learning | robot learning, manipulation, control |
| Embodied AI | embodied AI, humanoid, quadruped |

### Medical AI（医疗人工智能）

| 任务 | 关键词 |
|------|--------|
| Medical Imaging | CT, MRI, ultrasound, pathology |
| Cancer Detection | tumor detection, lesion segmentation |
| Medical LLM | medical LLM, medical VLM, clinical AI |
| Drug Discovery | drug discovery, protein folding |

---

## 环境变量

创建 `.env` 文件：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4
```

---

## 部署到 GitHub Pages

1. Fork 本仓库
2. Settings → Secrets → 添加 `LLM_API_KEY`
3. Settings → Pages → Source: GitHub Actions
4. 等待 Actions 自动运行

---

## 成本

| 功能 | 成本 |
|------|------|
| 基础版 | 免费 |
| OpenAlex API | 免费 |
| arXiv API | 免费 |
| GitHub 托管 | 免费 |
| GitHub Actions | 免费（2000分钟/月）|
| GLM API |按调用计费 |

---

## License

MIT
