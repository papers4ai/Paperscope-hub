"""
配置 - 论文分类
三大领域：世界模型、物理AI、医疗AI
"""

# ============================================================
# 搜索配置 - 分多个查询以获取更多论文
# ============================================================

# 分开搜索以避免URL过长和获取更多结果
# 每个查询都加了 cat: 类别过滤，确保爬取前就锁定目标领域，减少无关论文
# 格式：(关键词条件) AND (cat:领域1 OR cat:领域2 ...)
SEARCH_QUERIES = [
    # ── World Model ──────────────────────────────────────────────────────
    '(ti:"world model" OR abs:"world model") AND (cat:cs.LG OR cat:cs.CV OR cat:cs.AI OR cat:cs.RO)',
    '(ti:"video generation" OR ti:"video prediction" OR ti:"video diffusion") AND (cat:cs.CV OR cat:cs.LG)',
    '(ti:"neural radiance" OR ti:"gaussian splatting" OR ti:NeRF) AND cat:cs.CV',
    '(ti:"model-based reinforcement" OR ti:"model predictive control") AND (cat:cs.LG OR cat:cs.RO OR cat:cs.SY)',
    '(ti:"sim-to-real" OR ti:sim2real OR ti:"embodied agent") AND (cat:cs.RO OR cat:cs.LG OR cat:cs.AI)',

    # ── Physical AI ──────────────────────────────────────────────────────
    '(ti:"physics informed" OR ti:PINN OR ti:"physics-informed") AND (cat:cs.LG OR cat:cs.NA OR cat:cs.CE)',
    '(ti:"neural operator" OR ti:FNO OR ti:"deep operator" OR ti:"neural ODE") AND (cat:cs.LG OR cat:cs.NA)',
    '(ti:robot OR ti:robotics OR ti:manipulation OR ti:grasping) AND (cat:cs.RO OR cat:cs.LG)',
    '(ti:"embodied AI" OR ti:humanoid OR ti:quadruped OR ti:"dexterous") AND (cat:cs.RO OR cat:cs.AI OR cat:cs.LG)',
    '(ti:fluid OR ti:turbulence OR ti:"climate model" OR ti:atmospheric) AND (cat:cs.LG OR cat:cs.CE OR cat:physics.flu-dyn)',

    # ── Medical AI ───────────────────────────────────────────────────────
    '(ti:"medical imaging" OR ti:radiology OR ti:ultrasound OR ti:mammography) AND cat:cs.CV',
    '(ti:pathology OR ti:histopathology OR ti:"whole slide") AND (cat:cs.CV OR cat:cs.LG)',
    '(ti:"cancer detection" OR ti:"tumor detection" OR ti:lesion OR ti:nodule) AND cat:cs.CV',
    '(ti:"drug discovery" OR ti:"drug design" OR ti:"molecular generation") AND (cat:cs.LG OR cat:q-bio.QM OR cat:q-bio.BM)',
    '(ti:"protein folding" OR ti:alphafold OR ti:"protein structure" OR ti:"protein design") AND (cat:cs.LG OR cat:q-bio.BM)',
    '(ti:medical AND (ti:LLM OR ti:VLM OR ti:"language model" OR ti:"vision language")) AND (cat:cs.CV OR cat:cs.CL OR cat:cs.AI)',
]

# 兼容旧代码
SEARCH_QUERY = SEARCH_QUERIES[0]

# 增量更新窗口（天）：有历史数据时只抓最近 N 天
FETCH_RECENT_DAYS = 30

START_YEAR = 2023
END_YEAR = 2026  # 每次运行时会自动更新为当前年份
MAX_RESULTS = 50000  # arXiv API 上限
REQUEST_DELAY = 3

# ============================================================
# 三大领域定义
# ============================================================

DOMAINS = {
    "world_model": {
        "name": "World Model",
        "name_zh": "世界模型",
        "icon": "🌍",
        "color": "#6366f1",
        "description": "世界模型让AI理解物理世界，支持预测、规划和交互",
        "keywords": ["视频生成", "场景重建", "强化学习", "Sim2Real"]
    },
    "physical_ai": {
        "name": "Physical AI",
        "name_zh": "物理人工智能",
        "icon": "⚛️",
        "color": "#10b981",
        "description": "融合物理规律的深度学习，用于科学计算和智能系统",
        "keywords": ["PINN", "具身智能", "机器人", "流体模拟"]
    },
    "medical_ai": {
        "name": "Medical AI",
        "name_zh": "医疗人工智能",
        "icon": "🏥",
        "color": "#f43f5e",
        "description": "AI在医疗健康领域的应用，从影像到药物研发",
        "keywords": ["医学影像", "药物发现", "临床AI", "MedVLM"]
    }
}

# ============================================================
# 子领域/任务标签（按照计算机领域研究热点）
# ============================================================

TASK_DEFINITIONS = {
    # ==================== World Model ====================
    "WorldModel": ("世界模型核心", [
        r"world\s*model", r"world\s*representation", r"learned\s*simulator",
        r"internal\s*world", r"world\s*understanding", r"generative\s*world",
        r"actionable\s*model", r"world\s*action\s*model",
    ]),
    "ActionModel": ("动作/行为模型", [
        r"action\s*model", r"world\s*action", r"actionable\s*representation",
        r"action\s*prediction", r"action\s*conditioned", r"action\s*grounding",
        r"behavior\s*model", r"affordance", r"action\s*understanding",
    ]),
    "FoundationModel": ("基础大模型", [
        r"foundation\s*model", r"foundation\s*models",
        r"large\s*language\s*model", r"\bLLM\b", r"\bVLM\b",
        r"vision\s*language\s*model", r"multimodal\s*large",
        r"general\s*purpose\s*model", r"pretrained\s*model",
    ]),
    "VidGen": ("视频生成/预测", [
        r"video\s*generation", r"video\s*prediction", r"future\s*frame",
        r"frame\s*interpolation", r"motion\s*prediction", r"temporal\s*generation",
        r"video\s*diffusion", r"video\s*model", r"vdm", r"\bIVA\b",
        r"next\s*frame", r"video\s*synthesis", r"video\s*forecasting",
    ]),
    "NeRF": ("神经辐射场/3DGS", [
        r"neural\s*radiance", r"\bnerf\b", r"\bNeRF\b", r"gaussian\s*splatting",
        r"3dgs", r"novel\s*view", r"scene\s*reconstruction", r"view\s*synthesis",
        r"implicit\s*scene", r"volumetric\s*rendering",
        r"4d\s*gaussian", r"dynamic\s*gaussian", r"3d\s*generation",
    ]),
    "MBRL": ("基于模型的RL", [
        r"model[-\s]?based\s*reinforcement", r"model\s*predictive\s*control",
        r"\bMPC\b", r"dreamer", r"\bPlaNet\b", r"\bRSSM\b",
        r"latent\s*dynamics", r"planning\s*robot", r"model\s*learning",
        r"world\s*model\s*rl", r"learned\s*dynamics",
    ]),
    "Sim2Real": ("Sim-to-Real", [
        r"sim[-\s]?to[-\s]?real", r"sim2real", r"simulation",
        r"domain\s*randomization", r"simulated\s*environment",
        r"reality\s*gap", r"simulated\s*to\s*real",
        r"real\s*to\s*sim", r"real2sim",
    ]),
    "EmbodiedWM": ("具身世界模型", [
        r"embodied\s*world\s*model", r"visual\s*navigation",
        r"object\s*navigation", r"pointgoal", r"embodied\s*agent",
        r"interactive\s*perception", r"world\s*model\s*robot",
        r"embodied\s*planning", r"embodied\s*decision",
    ]),
    "Predictive": ("预测学习", [
        r"predictive\s*coding", r"predictive\s*model", r"forward\s*model",
        r"temporal\s*prediction", r"future\s*prediction",
        r"predictive\s*representation", r"masked\s*autoencoder",
        r"self\s*supervised\s*prediction", r"contrastive\s*prediction",
    ]),
    "DiffusionWM": ("扩散世界模型", [
        r"diffusion\s*world", r"diffusion\s*model",
        r"denoising\s*diffusion", r"ddpm", r"ddim",
        r"score\s*based\s*generative", r"diffusion\s*planning",
    ]),

    # ==================== Physical AI ====================
    "PINN": ("物理信息网络", [
        r"physics[-\s]?informed", r"\bPINN\b", r"\bPINNs\b",
        r"physics[-\s]?guided", r"physics[-\s]?constrained",
        r"physics[-\s]?regularized", r"neural\s*pde"
    ]),
    "NeuralOp": ("神经算子", [
        r"neural\s*operator", r"deep\s*onet", r"\bFNO\b",
        r"fourier\s*neural", r"operator\s*learning",
        r"\bDeepONet\b", r"neural\s*integral"
    ]),
    "Embodied": ("具身智能", [
        r"embodied\s*AI", r"embodied\s*intelligence", r"embodied\s*robot",
        r"humanoid", r"quadruped", r"mobile\s*robot"
    ]),
    "RobotLearn": ("机器人学习", [
        r"robot\s*learning", r"robotic\s*manipulation", r"robot\s*control",
        r"motion\s*planning", r"trajectory", r"imitation\s*learning",
        r"rl\s*robot", r"bimanual", r"dexterous\s*manipulation",
        r"grasp\s*detection", r"task\s*planning"
    ]),
    "FluidSim": ("流体/材料模拟", [
        r"fluid\s*dynamics", r"computational\s*fluid", r"turbulence",
        r"navier[-\s]?stokes", r"material\s*simulation",
        r"molecular\s*dynamics", r"particle\s*simulation"
    ]),
    "Climate": ("气候/天气预测", [
        r"climate\s*modeling", r"weather\s*prediction", r"atmospheric",
        r"earth\s*system", r"ocean\s*model", r"forecast",
        r"precipitation", r"cyclone"
    ]),
    "3DRecon": ("3D重建", [
        r"3d\s*reconstruction", r"depth\s*estimation", r"point\s*cloud",
        r"stereo\s*matching", r"multi[-\s]?view\s*3d",
        r"structure\s*from", r"sfm", r"mvs"
    ]),

    # ==================== Medical AI ====================
    "Pathology": ("病理AI", [
        r"pathology", r"pathological", r"histopathology", r"histological",
        r"whole\s*slide", r"\bwsi\b", r"cytology", r"cellular\s*analysis",
        r"gland\s*segmentation", r"tile\s*classification",
        r"breast\s*pathology", r"cancer\s*pathology", r"tumor\s*grading",
        r"Ki[-\s]?67", r"mitosis\s*detection", r"nuclei\s*segmentation",
        r"digital\s*pathology", r"computational\s*pathology"
    ]),
    "MedImg": ("医学影像AI", [
        r"medical\s*imaging", r"medical\s*image", r"segmentation",
        r"mri", r"magnetic\s*resonance", r"ct\s*scan", r"x[-\s]?ray",
        r"xray", r"ultrasound", r"ultrasonography",
        r"pet\s*scan", r"mammography", r"mammogram", r"retinal",
        r"fundus", r"optical\s*coherence", r"oct\s*scan",
        r"medical\s*image\s*segmentation", r"medical\s*image\s*classification"
    ]),
    "Cancer": ("癌症诊断", [
        r"cancer\s*detection", r"cancer\s*diagnosis", r"tumor\s*detection",
        r"tumor\s*segmentation", r"lesion\s*detection", r"nodule\s*detection",
        r"lung\s*cancer", r"breast\s*cancer", r"colorectal\s*cancer",
        r"gastric\s*cancer", r"hepatic\s*carcinoma", r"liver\s*cancer",
        r"prostate\s*cancer", r"thyroid\s*nodule", r"melanoma",
        r"pathology\s*cancer", r"oncology", r"malignancy\s*detection",
        r"abnormality\s*detection", r"microcalcification"
    ]),
    "MedVLM": ("医学多模态大模型", [
        r"medical\s*multimodal", r"medical\s*llm", r"medical\s*vlm",
        r"medical\s*vision[-\s]?language", r"medvlm", r"medical\s*vqa",
        r"medical\s*visual\s*question", r"radiology\s*vqa",
        r"clinical\s*vqa", r"medical\s*report\s*generation",
        r"medical\s*image\s*caption", r"chest\s*x[-\s]?ray\s*report",
        r"medical\s*conversation", r"clinical\s*multimodal"
    ]),
    "DrugMol": ("药物/分子设计", [
        r"drug\s*discovery", r"drug\s*design", r"molecular\s*generation",
        r"compound\s*screening", r"molecule\s*generation",
        r"de\s*novo\s*design", r"virtual\s*screening",
        r"molecular\s*docking", r"property\s*prediction"
    ]),
    "Protein": ("蛋白质/基因", [
        r"protein\s*folding", r"protein\s*structure", r"alphafold",
        r"esmfold", r"protein\s*design", r"protein\s*prediction",
        r"genomic", r"dna\s*sequence", r"rna\s*design"
    ]),
    "Clinical": ("临床决策支持", [
        r"clinical\s*decision", r"clinical\s*prediction", r"diagnosis",
        r"patient\s*outcome", r"electronic\s*health\s*record",
        r"\behr\b", r"medical\s*record", r"clinical\s*nlp"
    ]),
    "Surgery": ("手术/介入AI", [
        r"surgical\s*robot", r"computer[-\s]?aided\s*surgery",
        r"surgical\s*navigation", r"robotic\s*surgery",
        r"endoscopic", r"catheter", r"interventional"
    ]),
    "HealthMon": ("健康监测", [
        r"health\s*monitoring", r"wearable", r"vital\s*sign",
        r"mortality\s*prediction", r"icu", r"patient\s*monitor",
        r" physiological"
    ]),
}

# ============================================================
# 核心匹配关键词
# ============================================================

WORLD_MODEL_KEYWORDS = [
    # 核心概念
    r"world\s*model", r"\bWM\b", r"learned\s*simulator",
    r"internal\s*world", r"world\s*representation", r"world\s*understanding",
    # MBRL
    r"model[-\s]?based\s*reinforcement", r"model\s*predictive\s*control",
    r"latent\s*dynamics", r"\bMPC\b", r"dreamer", r"planet\b", r"rssm",
    # 视频
    r"video\s*generation", r"video\s*prediction", r"future\s*frame",
    r"frame\s*extrapolation", r"video\s*diffusion", r"vdm\b",
    # NeRF/3D
    r"neural\s*radiance", r"\bnerf\b", r"nerf\b", r"\bNeRF\b",
    r"gaussian\s*splatting", r"3dgs", r"novel\s*view\s*synthesis",
    r"scene\s*representation", r"view\s*synthesis",
    # Sim2Real
    r"sim[-\s]?to[-\s]?real", r"sim2real", r"domain\s*randomization",
    # 具身
    r"visual\s*navigation", r"object\s*navigation", r"pointgoal",
    r"embodied\s*agent", r"interactive\s*perception",
    # 扩散模型
    r"diffusion\s*world", r"world\s*model\s*diffusion", r"generative\s*world",
]

PHYSICAL_AI_KEYWORDS = [
    # PINN
    r"physics[-\s]?informed", r"\bPINN\b", r"\bPINNs\b",
    r"physics[-\s]?guided", r"physics[-\s]?constrained",
    r"variational\s*pin", r"neural\s*pde", r"pinn\b",
    # 神经算子
    r"neural\s*operator", r"deep\s*onet", r"\bFNO\b",
    r"fourier\s*neural\s*operator", r"operator\s*learning",
    r"learning\s*operator", r"\bONet\b",
    # 具身/机器人
    r"embodied\s*AI", r"embodied\s*intelligence", r"embodied\s*robot",
    r"robot\s*learning", r"robotic\s*manipulation", r"robot\s*control",
    r"motion\s*planning", r"trajectory\s*optimization",
    r"imitation\s*learning", r"rl\s*robot", r"bimanual",
    r"grasp\s*detection", r"dexterous",
    # 流体/物理
    r"fluid\s*dynamics", r"computational\s*fluid", r"navier[-\s]?stokes",
    r"turbulence", r"computational\s*physics", r"numerical\s*simulation",
    # 材料
    r"material\s*discovery", r"materials\s*design", r"computational\s*materials",
    r"molecular\s*dynamics",
    # 气候
    r"climate\s*modeling", r"weather\s*prediction", r"earth\s*system",
    r"atmospheric", r"ocean\s*model",
    # 物理推理
    r"physical\s*reasoning", r"physics\s*inference", r"dynamics\s*learning",
]

MEDICAL_AI_KEYWORDS = [
    # 通用医疗AI
    r"medical\s*AI", r"healthcare\s*AI", r"clinical\s*AI",
    r"computer[-\s]?aided\s*diagnosis", r"ai[-\s]?assisted\s*diagnosis",
    r"ai\s*for\s*health", r"health\s*AI",

    # ====== 病理AI (Pathology) ======
    r"pathology", r"pathological", r"histopathology", r"histological",
    r"whole\s*slide\s*image", r"\bwsi\b", r"cytology", r"cellular\s*pathology",
    r"digital\s*pathology", r"computational\s*pathology",
    r"breast\s*pathology", r"gland\s*segmentation",
    r"mitosis\s*detection", r"nuclei\s*detection", r"tile\s*classification",

    # ====== 医学影像 ======
    r"medical\s*imaging", r"medical\s*image", r"medical\s*image\s*analysis",
    r"x[-\s]?ray\s*analysis", r"ct\s*scan\b", r"mri\s*segmentation",
    r"brain\s*mri", r"chest\s*x[-\s]?ray", r"chest\s*radiograph",
    r"ultrasound\s*analysis", r"ultrasonography",
    r"mammography", r"mammogram", r"breast\s*screening",
    r"retinal\s*image", r"fundus\s*image", r"fundus", r"retinopathy",
    r"optical\s*coherence", r"\boct\b", r"cornea",
    r"segmentation\s*medical", r"medical\s*segmentation",

    # ====== 癌症检测 ======
    r"tumor\s*detection", r"cancer\s*detection", r"cancer\s*diagnosis",
    r"tumor\s*segmentation", r"lesion\s*detection", r"nodule\s*detection",
    r"lung\s*cancer", r"breast\s*cancer", r"colorectal\s*cancer",
    r"gastric\s*cancer", r"hepatic\s*carcinoma", r"liver\s*cancer",
    r"prostate\s*cancer", r"thyroid\s*nodule", r"melanoma",
    r"oncology", r"malignancy\s*detection", r"microcalcification",
    r"cancer\s*grading", r"gleason\s*score", r"Ki67",

    # ====== 医学多模态/大模型 ======
    r"medical\s*multimodal", r"medical\s*llm", r"medical\s*vlm",
    r"medical\s*vision[-\s]?language", r"vision[-\s]?language\s*medical",
    r"medvlm", r"medical\s*vqa", r"medical\s*visual\s*question",
    r"radiology\s*vqa", r"clinical\s*vqa", r"chest\s*x[-\s]?ray\s*vqa",
    r"medical\s*report\s*generation", r"medical\s*image\s*caption",
    r"medical\s*visual\s*reasoning", r"clinical\s*multimodal",

    # ====== 药物/分子 ======
    r"drug\s*discovery", r"drug\s*design", r"molecular\s*generation",
    r"compound\s*screening", r"virtual\s*screening", r"de\s*novo",
    r"molecular\s*property", r"binding\s*affinity",

    # ====== 蛋白质 ======
    r"protein\s*folding", r"protein\s*structure\s*prediction",
    r"\bAlphaFold\b", r"esmfold", r"protein\s*design",
    r"protein\s*language", r"protbert", r"genomic\s*analysis",

    # ====== 临床 ======
    r"clinical\s*decision", r"clinical\s*prediction", r"patient\s*outcome",
    r"electronic\s*health\s*record", r"\behr\b", r"diagnosis\s*support",
    r"clinical\s*nlp", r"medical\s*nlp",

    # ====== 手术机器人 ======
    r"surgical\s*robot", r"computer[-\s]?aided\s*surgery",
    r"surgical\s*navigation", r"robotic\s*surgery",
    r"endoscopic", r"laparoscopic",
    # 健康
    r"health\s*monitoring", r"wearable\s*device", r"vital\s*sign",
    r"mortality\s*prediction", r"icu\s*prediction", r"patient\s*monitor",
]

# ============================================================
# 知名模型
# ============================================================

# ============================================================
# 任务英文标签（用于动态生成 task_meta.json，供前端读取）
# ============================================================

TASK_EN_LABELS = {
    # World Model
    "WorldModel":     "World Model Core",
    "ActionModel":    "Action Model",
    "FoundationModel": "Foundation Model",
    "VidGen":         "Video Generation",
    "NeRF":           "NeRF / 3D Gaussian",
    "MBRL":           "Model-Based RL",
    "Sim2Real":       "Sim-to-Real",
    "EmbodiedWM":     "Embodied World Model",
    "Predictive":     "Predictive Learning",
    "DiffusionWM":    "Diffusion World Model",
    # Physical AI
    "PINN":       "Physics-Informed NN",
    "NeuralOp":   "Neural Operator",
    "Embodied":   "Embodied AI",
    "RobotLearn": "Robot Learning",
    "FluidSim":   "Fluid Simulation",
    "Climate":    "Climate Prediction",
    "3DRecon":    "3D Reconstruction",
    # Medical AI
    "Pathology":  "Pathology AI",
    "MedImg":     "Medical Imaging",
    "Cancer":     "Cancer Diagnosis",
    "MedVLM":     "Medical VLM",
    "DrugMol":    "Drug & Molecule",
    "Protein":    "Protein & Genome",
    "Clinical":   "Clinical Decision",
    "Surgery":    "Surgical AI",
    "HealthMon":  "Health Monitoring",
}

# 任务 → 所属领域（用于动态生成 domain_tasks 映射）
TASK_DOMAIN_MAP = {
    # World Model
    "WorldModel":     "world_model",
    "ActionModel":    "world_model",
    "FoundationModel": "world_model",
    "VidGen":         "world_model",
    "NeRF":           "world_model",
    "MBRL":           "world_model",
    "Sim2Real":       "world_model",
    "EmbodiedWM":     "world_model",
    "Predictive":     "world_model",
    "DiffusionWM":    "world_model",
    # Physical AI
    "PINN":       "physical_ai",
    "NeuralOp":   "physical_ai",
    "Embodied":   "physical_ai",
    "RobotLearn": "physical_ai",
    "FluidSim":   "physical_ai",
    "Climate":    "physical_ai",
    "3DRecon":    "physical_ai",
    # Medical AI
    "Pathology":  "medical_ai",
    "MedImg":     "medical_ai",
    "Cancer":     "medical_ai",
    "MedVLM":     "medical_ai",
    "DrugMol":    "medical_ai",
    "Protein":    "medical_ai",
    "Clinical":   "medical_ai",
    "Surgery":    "medical_ai",
    "HealthMon":  "medical_ai",
}

# ============================================================
# 知名模型
# ============================================================

MODEL_KEYWORDS = {
    "world_model": [
        r"\bDreamer\b", r"\bPlaNet\b", r"\bRSSM\b", r"\bGAE\b", r"\bSAE\b",
        r"\bCCIRS\b", r"S4\b", r"Kernelized\s*World\s*Model",
    ],
    "physical_ai": [
        r"DeepONet", r"\bFNO\b", r"\bPINN\b", r"\bONet\b",
        r"FNO\b", r"PINNet", r"VPINN",
        r"Chebychev", r"Spectral",
    ],
    "medical_ai": [
        r"\bMedGPT\b", r"\bMedPaLM\b", r"\bMedAlpaca\b", r"\bMedLM\b",
        r"\bAlphaFold\b", r"ESMFold", r"ESM\b",
        r"CheXpert", r"MIMIC", r"NIH\s*Chest",
        r"PathAI", r"GROVER\b",
    ]
}