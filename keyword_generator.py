"""
动态关键词生成器 - 使用 LLM 自动生成/更新搜索关键词

功能：
1. 基于领域定义自动生成关键词
2. 根据最新论文趋势更新关键词
3. 可手动编辑关键词配置
"""

import json
import os
import logging
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)

# 关键词配置文件路径
KEYWORDS_FILE = "keywords.json"

# 领域定义（用于提示 LLM）
DOMAIN_DESCRIPTIONS = {
    "world_model": {
        "name": "World Model",
        "name_zh": "世界模型",
        "description": """
World Model 是让 AI 理解并模拟物理世界的模型。核心研究方向包括：
- 世界模型架构：如何学习世界表征、latent dynamics
- 动作模型：action model, world action model, affordance learning
- 视频生成/预测：video generation, video prediction, future prediction
- 3D 场景理解：NeRF, Gaussian Splatting, 3D reconstruction
- 基于模型的 RL：model-based RL, planning, decision making
- 具身智能：embodied AI, robot navigation, interaction
- 扩散模型：diffusion models for world modeling

近期热点：
- World Action Model
- Foundation Model 在具身智能的应用
- Video diffusion models
- 3D Gaussian Splatting
""",
    },
    "physical_ai": {
        "name": "Physical AI",
        "name_zh": "物理人工智能",
        "description": """
Physical AI 是融合物理规律与深度学习的交叉领域。核心研究方向：
- 物理信息网络：PINN, physics-informed neural networks
- 神经算子：Neural Operator, FNO, DeepONet
- 机器人学习：robot learning, manipulation, control
- 具身智能：embodied AI, humanoid, quadruped
- 流体/材料模拟：CFD, turbulence, material simulation
- 气候/天气：climate modeling, weather prediction

近期热点：
- Foundation model for robotics
- Differentiable simulation
- Neural surrogate models
""",
    },
    "medical_ai": {
        "name": "Medical AI",
        "name_zh": "医疗人工智能",
        "description": """
Medical AI 是 AI 在医疗健康领域的应用。核心研究方向：
- 医学影像：CT, MRI, X-ray, ultrasound, pathology
- 癌症诊断：detection, segmentation, classification
- 医学大模型：medical LLM, medical VLM, clinical AI
- 药物发现：drug discovery, molecular generation
- 蛋白质：protein folding, structure prediction

近期热点：
- Multimodal medical AI
- Foundation models in healthcare
- AI-assisted diagnosis
- Clinical decision support
""",
    },
}


def load_keywords() -> Dict:
    """加载关键词配置"""
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE) as f:
            return json.load(f)

    # 默认关键词
    return {
        "world_model": [
            "world model", "world action model", "actionable model",
            "video generation", "video prediction", "video diffusion",
            "NeRF", "gaussian splatting", "model-based RL",
            "embodied AI", "diffusion model",
        ],
        "physical_ai": [
            "physics-informed neural network", "PINN", "neural operator",
            "FNO", "robot learning", "embodied AI",
            "fluid dynamics", "climate modeling",
        ],
        "medical_ai": [
            "medical imaging AI", "pathology AI", "cancer detection",
            "medical LLM", "drug discovery", "protein folding",
        ],
    }


def save_keywords(keywords: Dict):
    """保存关键词配置"""
    with open(KEYWORDS_FILE, "w") as f:
        json.dump(keywords, f, indent=2)
    logger.info(f"Saved keywords to {KEYWORDS_FILE}")


def generate_keywords_with_llm(
    domain: str,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    existing_keywords: List[str] = None,
    num_keywords: int = 20,
) -> List[str]:
    """
    使用 LLM 生成关键词

    Args:
        domain: 领域名称 (world_model, physical_ai, medical_ai)
        api_key: LLM API key
        base_url: LLM API base URL
        model: 模型名称
        existing_keywords: 已有关键词（供 LLM 参考）
        num_keywords: 生成数量
    """
    if domain not in DOMAIN_DESCRIPTIONS:
        logger.error(f"Unknown domain: {domain}")
        return []

    domain_info = DOMAIN_DESCRIPTIONS[domain]

    prompt = f"""You are an expert researcher in AI/ML field.

Generate {num_keywords} search keywords for finding relevant papers on arXiv.

Domain: {domain_info['name']} ({domain_info['name_zh']})

Description:
{domain_info['description']}

Existing keywords (for reference): {existing_keywords}

Requirements:
1. Keywords should be specific enough to find relevant papers
2. Include both broad terms (like "world model") and specific techniques (like "Gaussian Splatting")
3. Include recent hot topics and emerging research directions
4. Prioritize keywords that would appear in paper titles
5. Return ONLY a JSON list of keywords, no explanation

Example format: ["keyword1", "keyword2", "keyword3"]
"""

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model or "glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # 解析 JSON
        # 尝试提取 JSON 列表
        if "[" in content and "]" in content:
            start = content.index("[")
            end = content.rindex("]") + 1
            keywords = json.loads(content[start:end])
            return keywords

        return []

    except Exception as e:
        logger.error(f"LLM keyword generation failed: {e}")
        return []


def update_keywords(
    api_key: str = None,
    base_url: str = None,
    model: str = None,
):
    """
    更新所有领域的关键词

    读取环境变量 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    """
    # 从环境变量获取配置
    api_key = api_key or os.environ.get("LLM_API_KEY")
    base_url = base_url or os.environ.get("LLM_BASE_URL")
    model = model or os.environ.get("LLM_MODEL", "glm-4")

    if not api_key:
        logger.error("LLM_API_KEY not set")
        return

    keywords = load_keywords()

    logger.info("Updating keywords with LLM...")

    for domain in DOMAIN_DESCRIPTIONS:
        logger.info(f"\n=== Updating {domain} ===")

        existing = keywords.get(domain, [])
        new_keywords = generate_keywords_with_llm(
            domain=domain,
            api_key=api_key,
            base_url=base_url,
            model=model,
            existing_keywords=existing,
        )

        if new_keywords:
            # 合并去重
            all_keywords = list(set(existing + new_keywords))
            keywords[domain] = all_keywords
            logger.info(f"Generated {len(new_keywords)} new keywords, total: {len(all_keywords)}")

    save_keywords(keywords)
    return keywords


def get_all_keywords() -> List[str]:
    """获取所有关键词（用于搜索）"""
    keywords = load_keywords()
    all_kw = []
    for domain_kws in keywords.values():
        all_kw.extend(domain_kws)
    return list(set(all_kw))


def get_openalex_keywords() -> List[str]:
    """
    获取 OpenAlex 搜索关键词

    优先级：
    1. keywords.json（如果存在）
    2. 默认关键词
    """
    keywords = load_keywords()

    # 展平为列表
    all_keywords = []
    for domain_kws in keywords.values():
        all_keywords.extend(domain_kws)

    return list(set(all_keywords))


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="关键词生成器")
    parser.add_argument("--update", action="store_true", help="使用 LLM 更新关键词")
    parser.add_argument("--show", action="store_true", help="显示当前关键词")
    args = parser.parse_args()

    if args.update:
        keywords = update_keywords()
        if keywords:
            print("\n更新后的关键词：")
            for domain, kws in keywords.items():
                print(f"\n{domain}:")
                for kw in kws:
                    print(f"  - {kw}")

    if args.show:
        keywords = load_keywords()
        print("\n当前关键词：")
        for domain, kws in keywords.items():
            print(f"\n{domain}: {len(kws)} 个")
            for kw in kws[:10]:
                print(f"  - {kw}")
            if len(kws) > 10:
                print(f"  ... 共 {len(kws)} 个")

    if not args.update and not args.show:
        # 默认：显示
        keywords = load_keywords()
        print("\n当前关键词：")
        for domain, kws in keywords.items():
            print(f"\n{domain}: {len(kws)} 个")
