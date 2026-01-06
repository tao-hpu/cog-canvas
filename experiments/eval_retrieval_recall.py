#!/usr/bin/env python3
"""
检索召回率评估实验

目的：评估检索阶段的质量，不涉及 LLM 回答
证明 Graph 的贡献来自检索质量提升，而非 LLM 能力

指标：
- Retrieval Recall: 检索结果覆盖多少 ground truth 关键词
- Retrieval Precision: 检索结果中有多少是相关的
- Contains GT: 检索结果是否包含 ground truth 答案

用法：
    python -m experiments.eval_retrieval_recall --agent cogcanvas --samples 10 -vv
    python -m experiments.eval_retrieval_recall --agent cogcanvas-no-graph --samples 10 -vv
"""

import argparse
import json
import re
import string
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import defaultdict

# 复用现有的评分函数
from experiments.runner_locomo import (
    normalize_answer,
    tokenize_and_stem,
    get_extraction_config_hash,
    get_cache_path,
)
from experiments.locomo_adapter import (
    load_locomo,
    convert_to_eval_format,
    LoCoMoConversation,
    LoCoMoQAPair,
)


@dataclass
class RetrievalRecallResult:
    """单个问题的检索召回率结果"""
    question: str
    category: int  # 1=单跳, 2=时间, 3=多跳
    ground_truth: str

    # 关键词分析
    gt_keywords: List[str]
    retrieved_keywords: List[str]
    common_keywords: List[str]

    # 指标
    retrieval_recall: float  # common / gt
    retrieval_precision: float  # common / retrieved
    retrieval_f1: float
    contains_gt: bool  # 是否包含完整答案

    # 检索详情
    num_objects_retrieved: int
    retrieved_content_preview: str  # 前 200 字符


@dataclass
class ConversationRetrievalResult:
    """单个对话的检索召回率结果"""
    conversation_id: str
    num_questions: int
    qa_results: List[RetrievalRecallResult]

    # 汇总
    avg_recall: float
    avg_precision: float
    avg_f1: float
    contains_gt_rate: float


def extract_keywords(text: str) -> List[str]:
    """提取关键词（使用现有的标准化逻辑）"""
    normalized = normalize_answer(text)
    return tokenize_and_stem(normalized)


def compute_retrieval_metrics(
    gt_keywords: List[str],
    retrieved_keywords: List[str],
) -> tuple:
    """计算检索指标"""
    gt_set = set(gt_keywords)
    retrieved_set = set(retrieved_keywords)
    common = gt_set & retrieved_set

    recall = len(common) / len(gt_set) if gt_set else 0.0
    precision = len(common) / len(retrieved_set) if retrieved_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return recall, precision, f1, list(common)


def check_contains_gt(retrieved_text: str, ground_truth: str) -> bool:
    """检查检索结果是否包含完整的 ground truth"""
    retrieved_normalized = normalize_answer(retrieved_text)
    gt_normalized = normalize_answer(ground_truth)
    return gt_normalized in retrieved_normalized


def evaluate_single_question(
    agent,
    question: str,
    ground_truth: str,
    category: int,
    top_k: int = 10,
) -> RetrievalRecallResult:
    """评估单个问题的检索召回率"""

    # 1. 执行检索（不调用 LLM 回答）
    canvas = agent._canvas

    # 使用 agent 的配置
    retrieval_result = canvas.retrieve(
        query=question,
        top_k=top_k,
        method=agent.retrieval_method,
        include_related=agent.enable_graph_expansion,
        max_hops=agent.graph_hops,
    )

    # 2. 组合检索内容
    retrieved_parts = []
    for obj in retrieval_result.objects:
        if obj.quote:
            retrieved_parts.append(obj.quote)
        if obj.content:
            retrieved_parts.append(obj.content)
    retrieved_text = " ".join(retrieved_parts)

    # 3. 提取关键词
    gt_keywords = extract_keywords(ground_truth)
    retrieved_keywords = extract_keywords(retrieved_text)

    # 4. 计算指标
    recall, precision, f1, common = compute_retrieval_metrics(gt_keywords, retrieved_keywords)
    contains_gt = check_contains_gt(retrieved_text, ground_truth)

    return RetrievalRecallResult(
        question=question,
        category=category,
        ground_truth=ground_truth,
        gt_keywords=gt_keywords,
        retrieved_keywords=retrieved_keywords[:50],  # 限制长度
        common_keywords=common,
        retrieval_recall=recall,
        retrieval_precision=precision,
        retrieval_f1=f1,
        contains_gt=contains_gt,
        num_objects_retrieved=len(retrieval_result.objects),
        retrieved_content_preview=retrieved_text[:200] + "..." if len(retrieved_text) > 200 else retrieved_text,
    )


def evaluate_conversation(
    agent,
    conversation: LoCoMoConversation,
    top_k: int = 10,
    categories: List[int] = [1, 2, 3],
    verbose: int = 0,
) -> ConversationRetrievalResult:
    """评估单个对话的所有问题"""

    qa_results = []

    for qa in conversation.qa_pairs:
        if qa.category not in categories:
            continue

        result = evaluate_single_question(
            agent=agent,
            question=qa.question,
            ground_truth=qa.answer,
            category=qa.category,
            top_k=top_k,
        )
        qa_results.append(result)

        if verbose >= 2:
            cat_name = {1: "1-hop", 2: "temporal", 3: "multi-hop"}.get(qa.category, "?")
            print(f"  [{cat_name}] Recall={result.retrieval_recall:.2f} ContainsGT={result.contains_gt}")

    # 汇总
    if qa_results:
        avg_recall = sum(r.retrieval_recall for r in qa_results) / len(qa_results)
        avg_precision = sum(r.retrieval_precision for r in qa_results) / len(qa_results)
        avg_f1 = sum(r.retrieval_f1 for r in qa_results) / len(qa_results)
        contains_gt_rate = sum(1 for r in qa_results if r.contains_gt) / len(qa_results)
    else:
        avg_recall = avg_precision = avg_f1 = contains_gt_rate = 0.0

    return ConversationRetrievalResult(
        conversation_id=conversation.id,
        num_questions=len(qa_results),
        qa_results=qa_results,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        avg_f1=avg_f1,
        contains_gt_rate=contains_gt_rate,
    )


def run_experiment(
    agent_name: str,
    samples: Optional[int] = None,
    top_k: int = 10,
    categories: List[int] = [1, 2, 3],
    verbose: int = 0,
    load_cache: bool = True,
):
    """运行检索召回率实验"""

    print(f"\n{'='*60}")
    print(f"RETRIEVAL RECALL EXPERIMENT")
    print(f"Agent: {agent_name}")
    print(f"{'='*60}\n")

    # 加载数据集
    dataset_path = Path(__file__).parent / "data" / "locomo10.json"
    raw_data = load_locomo(str(dataset_path))
    all_conversations = convert_to_eval_format(raw_data)
    conversations = all_conversations[:samples] if samples else all_conversations

    print(f"Loaded {len(conversations)} conversations")

    # 创建 agent
    from experiments.agents.cogcanvas_agent import CogCanvasAgent

    # 复用 runner_locomo.py 中的配置逻辑
    config = {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": top_k,
        "graph_hops": 3,
        "use_reranker": True,
        "reranker_candidate_k": 20,
    }

    # 应用 agent 变体配置
    if agent_name in ("cogcanvas-nograph", "cogcanvas-no-graph"):
        config["enable_graph_expansion"] = False
    elif agent_name == "cogcanvas-no-cot":
        config["prompt_style"] = "direct"
    elif agent_name == "cogcanvas-no-temporal":
        config["enable_temporal_heuristic"] = False
    elif agent_name == "cogcanvas-no-hybrid":
        config["retrieval_method"] = "semantic"
    elif agent_name == "cogcanvas-no-rerank":
        config["use_reranker"] = False
    elif agent_name == "cogcanvas-minimal":
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": False,
            "retrieval_method": "semantic",
            "prompt_style": "direct",
            "retrieval_top_k": top_k,
            "graph_hops": 1,
            "use_reranker": False,
        }

    agent = CogCanvasAgent(**config)

    # 评估每个对话
    all_results = []
    category_stats = defaultdict(lambda: {"recall": [], "precision": [], "f1": [], "contains_gt": []})

    for i, conv in enumerate(conversations):
        if verbose >= 1:
            print(f"\n[{i+1}/{len(conversations)}] {conv.id}")

        # 重置 agent
        agent.reset()

        # 尝试加载缓存
        config_hash = get_extraction_config_hash(config)
        cache_path = get_cache_path(conv.id, config_hash)

        if load_cache and cache_path.exists():
            agent.load_canvas_state(str(cache_path))
            if verbose >= 1:
                print(f"  Loaded from cache")
        else:
            # 需要先处理对话以构建 Canvas
            for turn in conv.turns:
                agent.process_turn(turn)
            if verbose >= 1:
                print(f"  Processed {len(conv.turns)} turns")

        # 评估
        conv_result = evaluate_conversation(
            agent=agent,
            conversation=conv,
            top_k=top_k,
            categories=categories,
            verbose=verbose,
        )
        all_results.append(conv_result)

        # 按类别统计
        for qa_result in conv_result.qa_results:
            cat = qa_result.category
            category_stats[cat]["recall"].append(qa_result.retrieval_recall)
            category_stats[cat]["precision"].append(qa_result.retrieval_precision)
            category_stats[cat]["f1"].append(qa_result.retrieval_f1)
            category_stats[cat]["contains_gt"].append(1 if qa_result.contains_gt else 0)

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"RETRIEVAL RECALL RESULTS")
    print(f"{'='*60}")
    print(f"Agent: {agent_name}")
    print(f"Conversations: {len(conversations)}")
    print(f"Top-k: {top_k}")
    print(f"Graph Expansion: {config.get('enable_graph_expansion', True)}")
    print(f"Retrieval Method: {config.get('retrieval_method', 'hybrid')}")
    print()

    # 总体指标
    all_recalls = [r for stats in category_stats.values() for r in stats["recall"]]
    all_precisions = [r for stats in category_stats.values() for r in stats["precision"]]
    all_f1s = [r for stats in category_stats.values() for r in stats["f1"]]
    all_contains = [r for stats in category_stats.values() for r in stats["contains_gt"]]

    print(f"Overall:")
    print(f"  Retrieval Recall:    {sum(all_recalls)/len(all_recalls)*100:.1f}%")
    print(f"  Retrieval Precision: {sum(all_precisions)/len(all_precisions)*100:.1f}%")
    print(f"  Retrieval F1:        {sum(all_f1s)/len(all_f1s)*100:.1f}%")
    print(f"  Contains GT Rate:    {sum(all_contains)/len(all_contains)*100:.1f}%")
    print()

    # 按类别
    cat_names = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop"}
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        if not stats["recall"]:
            continue
        print(f"{cat_names.get(cat, f'Cat-{cat}')}:")
        print(f"  Recall:      {sum(stats['recall'])/len(stats['recall'])*100:.1f}%")
        print(f"  Precision:   {sum(stats['precision'])/len(stats['precision'])*100:.1f}%")
        print(f"  F1:          {sum(stats['f1'])/len(stats['f1'])*100:.1f}%")
        print(f"  Contains GT: {sum(stats['contains_gt'])/len(stats['contains_gt'])*100:.1f}%")
        print()

    # 保存结果
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"retrieval_recall_{agent_name}.json"

    summary = {
        "agent": agent_name,
        "config": config,
        "num_conversations": len(conversations),
        "top_k": top_k,
        "overall": {
            "retrieval_recall": sum(all_recalls)/len(all_recalls),
            "retrieval_precision": sum(all_precisions)/len(all_precisions),
            "retrieval_f1": sum(all_f1s)/len(all_f1s),
            "contains_gt_rate": sum(all_contains)/len(all_contains),
        },
        "by_category": {
            cat_names.get(cat, f"cat_{cat}"): {
                "retrieval_recall": sum(stats["recall"])/len(stats["recall"]) if stats["recall"] else 0,
                "retrieval_precision": sum(stats["precision"])/len(stats["precision"]) if stats["precision"] else 0,
                "retrieval_f1": sum(stats["f1"])/len(stats["f1"]) if stats["f1"] else 0,
                "contains_gt_rate": sum(stats["contains_gt"])/len(stats["contains_gt"]) if stats["contains_gt"] else 0,
            }
            for cat, stats in category_stats.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="检索召回率评估实验")
    parser.add_argument(
        "--agent", "-a",
        choices=[
            "cogcanvas",
            "cogcanvas-no-graph",
            "cogcanvas-nograph",
            "cogcanvas-no-cot",
            "cogcanvas-no-temporal",
            "cogcanvas-no-hybrid",
            "cogcanvas-no-rerank",
            "cogcanvas-minimal",
        ],
        default="cogcanvas",
        help="Agent to evaluate",
    )
    parser.add_argument("--samples", "-n", type=int, default=None, help="Number of conversations")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Top-k retrieval")
    parser.add_argument("--categories", "-c", type=int, nargs="+", default=[1, 2, 3], help="Question categories")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity level")
    parser.add_argument("--no-cache", action="store_true", help="Don't load from cache")

    args = parser.parse_args()

    run_experiment(
        agent_name=args.agent,
        samples=args.samples,
        top_k=args.top_k,
        categories=args.categories,
        verbose=args.verbose,
        load_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
