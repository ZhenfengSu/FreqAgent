import json
import re
import os
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
from copy import deepcopy

# ============ 配置 ============
SAMPLING_K = 20  # Monte Carlo 采样次数
SAMPLING_TEMPERATURE = 0.7  # 采样温度
SAMPLING_TOP_P = 0.95

# API 配置
INFERENCE_API_BASE = "http://127.0.0.1:6001/v1"
INFERENCE_API_KEY = "EMPTY"
INFERENCE_MODEL = "your-model-name"  # 修改为你的模型名

JUDGE_API_BASE = "http://127.0.0.1:6002/v1"
JUDGE_API_KEY = "EMPTY"
JUDGE_MODEL = "qwen2.5-72b-instruct"

# Final Answer Probe 的 prompt
FINAL_ANSWER_PROBE_PROMPT = """Based on all the information and context above, please now provide your best answer to the original question. Think step by step and give your final answer in the following format:
<think>your reasoning</think>
<answer>your answer</answer>"""

# Judge prompt - 用于判断两个答案是否等价
ANSWER_EQUIVALENCE_PROMPT = """You are a judge that determines if two answers are semantically equivalent for the given question.

Question: {question}
Answer 1: {answer1}
Answer 2: {answer2}

Are these two answers semantically equivalent (answering the same thing, possibly with different wording)?
Reply with only: equivalent: yes or equivalent: no"""


@dataclass
class RoundData:
    """单轮数据"""
    round_id: int
    context_messages: List[Dict]  # 截止到该轮的上下文
    sampled_answers: List[str]    # K个采样答案
    answer_clusters: Dict[str, int]  # 答案聚类及频次
    entropy: float                # H(A_t)
    information_gain: float       # IG_t = H(A_{t-1}) - H(A_t)


@dataclass 
class QuestionIGData:
    """单个问题的IG数据"""
    question: str
    ground_truth_answer: str
    rollout_id: str
    total_rounds: int
    rounds: List[RoundData]
    final_prediction: str
    termination: str


class IGCalculator:
    def __init__(
        self,
        inference_api_base: str = INFERENCE_API_BASE,
        inference_api_key: str = INFERENCE_API_KEY,
        inference_model: str = INFERENCE_MODEL,
        judge_api_base: str = JUDGE_API_BASE,
        judge_api_key: str = JUDGE_API_KEY,
        judge_model: str = JUDGE_MODEL,
        sampling_k: int = SAMPLING_K,
        sampling_temperature: float = SAMPLING_TEMPERATURE,
        sampling_top_p: float = SAMPLING_TOP_P,
    ):
        self.inference_client = OpenAI(
            api_key=inference_api_key,
            base_url=inference_api_base,
        )
        self.inference_model = inference_model
        
        self.judge_client = OpenAI(
            api_key=judge_api_key,
            base_url=judge_api_base,
        )
        self.judge_model = judge_model
        
        self.sampling_k = sampling_k
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
    
    def call_inference(self, messages: List[Dict], max_tries: int = 5) -> Optional[str]:
        """调用推理模型"""
        for attempt in range(max_tries):
            try:
                response = self.inference_client.chat.completions.create(
                    model=self.inference_model,
                    messages=messages,
                    temperature=self.sampling_temperature,
                    top_p=self.sampling_top_p,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == max_tries - 1:
                    print(f"Inference error: {e}")
                    return None
        return None
    
    def call_judge(self, prompt: str, max_tries: int = 5) -> Optional[str]:
        """调用 judge 模型"""
        for attempt in range(max_tries):
            try:
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == max_tries - 1:
                    print(f"Judge error: {e}")
                    return None
        return None
    
    def extract_answer(self, content: str) -> str:
        """从模型输出中提取答案"""
        if '<answer>' in content and '</answer>' in content:
            answer = content.split('<answer>')[1].split('</answer>')[0]
            return answer.strip()
        # 如果没有标准格式，返回整个内容（简化处理）
        return content.strip()
    
    def probe_final_answer(self, context_messages: List[Dict]) -> Optional[str]:
        """
        FinalAnswerProbe: 在给定上下文后，询问模型当前的最终答案
        """
        # 复制上下文并添加 probe prompt
        messages = deepcopy(context_messages)
        messages.append({
            "role": "user",
            "content": FINAL_ANSWER_PROBE_PROMPT
        })
        
        response = self.call_inference(messages)
        if response:
            return self.extract_answer(response)
        return None
    
    def sample_answers(self, context_messages: List[Dict], k: int = None) -> List[str]:
        """
        Monte Carlo Sampling: 采样 K 个最终答案
        """
        if k is None:
            k = self.sampling_k
        
        answers = []
        for _ in range(k):
            answer = self.probe_final_answer(context_messages)
            if answer:
                answers.append(answer)
        
        return answers
    
    def check_answer_equivalence(self, question: str, answer1: str, answer2: str) -> bool:
        """
        使用 LLM-as-judge 判断两个答案是否等价
        """
        # 先做简单的 exact match
        if self.normalize_answer(answer1) == self.normalize_answer(answer2):
            return True
        
        # 使用 LLM judge
        prompt = ANSWER_EQUIVALENCE_PROMPT.format(
            question=question,
            answer1=answer1,
            answer2=answer2
        )
        
        response = self.call_judge(prompt)
        if response:
            match = re.search(r'equivalent\s*:\s*(yes|no)', response, re.IGNORECASE)
            if match:
                return match.group(1).lower() == 'yes'
        
        # 默认不等价
        return False
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """标准化答案用于比较"""
        # 转小写
        answer = answer.lower()
        # 移除多余空白
        answer = ' '.join(answer.split())
        # 移除标点（可选）
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer.strip()
    
    def cluster_answers(self, question: str, answers: List[str], use_llm_judge: bool = True) -> Dict[str, int]:
        """
        将答案聚类到等价类
        返回: {代表答案: 频次}
        """
        if not answers:
            return {}
        
        clusters = []  # List of (representative_answer, count)
        
        for answer in answers:
            found_cluster = False
            
            for i, (rep_answer, count) in enumerate(clusters):
                if use_llm_judge:
                    is_equiv = self.check_answer_equivalence(question, answer, rep_answer)
                else:
                    is_equiv = self.normalize_answer(answer) == self.normalize_answer(rep_answer)
                
                if is_equiv:
                    clusters[i] = (rep_answer, count + 1)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append((answer, 1))
        
        return {rep: count for rep, count in clusters}
    
    def cluster_answers_simple(self, answers: List[str]) -> Dict[str, int]:
        """
        简单聚类：基于标准化后的 exact match
        速度快，适合快速实验
        """
        normalized_to_original = {}
        counts = Counter()
        
        for answer in answers:
            norm = self.normalize_answer(answer)
            if norm not in normalized_to_original:
                normalized_to_original[norm] = answer
            counts[norm] += 1
        
        return {normalized_to_original[norm]: count for norm, count in counts.items()}
    
    @staticmethod
    def calculate_entropy(cluster_counts: Dict[str, int]) -> float:
        """
        计算熵 H = -sum(p_i * log(p_i))
        """
        if not cluster_counts:
            return 0.0
        
        total = sum(cluster_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in cluster_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def parse_rounds_from_messages(self, messages: List[Dict]) -> List[List[Dict]]:
        """
        从 messages 中解析每一轮的上下文
        一轮定义为：到 tool_response 结束
        返回：每轮结束时的完整上下文列表
        """
        rounds = []
        current_round_end_indices = []
        
        for i, msg in enumerate(messages):
            # 检查是否是 tool_response（标志一轮结束）
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if '<tool_response>' in content or content.startswith('<tool_response>'):
                    current_round_end_indices.append(i)
            
            # 或者检查 assistant 给出最终答案
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if '<answer>' in content and '</answer>' in content:
                    # 最后一轮
                    current_round_end_indices.append(i)
        
        # 构建每轮的上下文
        for end_idx in current_round_end_indices:
            # 上下文包含从开始到该轮结束的所有消息
            context = messages[:end_idx + 1]
            rounds.append(context)
        
        # 如果没有找到任何轮次，至少返回完整的 messages
        if not rounds and messages:
            rounds.append(messages)
        
        return rounds
    
    def process_single_question(
        self, 
        item: Dict, 
        use_llm_judge: bool = False,
        verbose: bool = True
    ) -> QuestionIGData:
        """
        处理单个问题，计算每一轮的 IG
        
        Args:
            item: jsonl 中的一条数据
            use_llm_judge: 是否使用 LLM 判断答案等价（更准确但更慢）
            verbose: 是否打印详细信息
        """
        # 提取基本信息
        question = item.get('question', '')
        answer = item.get('answer', '')
        rollout_id = item.get('rollout_id', '')
        messages = item.get('messages', [])
        prediction = item.get('prediction', '')
        termination = item.get('termination', '')
        
        # 解析每一轮
        round_contexts = self.parse_rounds_from_messages(messages)
        total_rounds = len(round_contexts)
        
        if verbose:
            print(f"\nProcessing question: {question[:50]}...")
            print(f"Total rounds: {total_rounds}")
        
        rounds_data = []
        prev_entropy = None
        
        for round_id, context in enumerate(round_contexts):
            if verbose:
                print(f"  Round {round_id + 1}/{total_rounds}: Sampling {self.sampling_k} answers...")
            
            # Monte Carlo Sampling
            sampled_answers = self.sample_answers(context, k=self.sampling_k)
            
            if verbose:
                print(f"    Got {len(sampled_answers)} valid answers")
            
            # 聚类答案
            if use_llm_judge:
                clusters = self.cluster_answers(question, sampled_answers, use_llm_judge=True)
            else:
                clusters = self.cluster_answers_simple(sampled_answers)
            
            if verbose:
                print(f"    Clusters: {len(clusters)}")
            
            # 计算熵
            entropy = self.calculate_entropy(clusters)
            
            # 计算 IG
            if prev_entropy is None:
                ig = 0.0  # 第一轮没有 IG
            else:
                ig = prev_entropy - entropy
            
            if verbose:
                print(f"    Entropy: {entropy:.4f}, IG: {ig:.4f}")
            
            round_data = RoundData(
                round_id=round_id + 1,
                context_messages=context,  # 可以选择不保存完整上下文以节省空间
                sampled_answers=sampled_answers,
                answer_clusters=clusters,
                entropy=entropy,
                information_gain=ig
            )
            rounds_data.append(round_data)
            
            prev_entropy = entropy
        
        return QuestionIGData(
            question=question,
            ground_truth_answer=answer,
            rollout_id=rollout_id,
            total_rounds=total_rounds,
            rounds=rounds_data,
            final_prediction=prediction,
            termination=termination
        )


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_results(results: List[QuestionIGData], output_path: str, save_context: bool = False):
    """
    保存结果到 JSON 文件
    
    Args:
        results: 处理结果列表
        output_path: 输出路径
        save_context: 是否保存完整上下文（会显著增大文件大小）
    """
    output_data = []
    
    for result in results:
        item = {
            "question": result.question,
            "ground_truth_answer": result.ground_truth_answer,
            "rollout_id": result.rollout_id,
            "total_rounds": result.total_rounds,
            "final_prediction": result.final_prediction,
            "termination": result.termination,
            "rounds": []
        }
        
        for round_data in result.rounds:
            round_item = {
                "round_id": round_data.round_id,
                "sampled_answers": round_data.sampled_answers,
                "answer_clusters": round_data.answer_clusters,
                "num_clusters": len(round_data.answer_clusters),
                "entropy": round_data.entropy,
                "information_gain": round_data.information_gain,
            }
            
            if save_context:
                round_item["context_messages"] = round_data.context_messages
            
            item["rounds"].append(round_item)
        
        # 计算汇总统计
        if result.rounds:
            item["summary"] = {
                "initial_entropy": result.rounds[0].entropy,
                "final_entropy": result.rounds[-1].entropy,
                "total_ig": result.rounds[0].entropy - result.rounds[-1].entropy if len(result.rounds) > 1 else 0,
                "avg_ig_per_round": np.mean([r.information_gain for r in result.rounds[1:]]) if len(result.rounds) > 1 else 0,
                "max_ig_round": max(range(len(result.rounds)), key=lambda i: result.rounds[i].information_gain) + 1 if result.rounds else 0,
                "entropy_trajectory": [r.entropy for r in result.rounds],
                "ig_trajectory": [r.information_gain for r in result.rounds],
            }
        
        output_data.append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")


def save_summary_statistics(results: List[QuestionIGData], output_path: str):
    """保存汇总统计信息"""
    summary = {
        "total_questions": len(results),
        "total_rounds": sum(r.total_rounds for r in results),
        "avg_rounds_per_question": np.mean([r.total_rounds for r in results]),
        "questions_with_answer": sum(1 for r in results if r.termination == 'answer'),
    }
    
    # 收集所有轮次的 IG 数据
    all_igs = []
    all_entropies = []
    
    for result in results:
        for round_data in result.rounds:
            all_igs.append(round_data.information_gain)
            all_entropies.append(round_data.entropy)
    
    if all_igs:
        summary["ig_statistics"] = {
            "mean": float(np.mean(all_igs)),
            "std": float(np.std(all_igs)),
            "min": float(np.min(all_igs)),
            "max": float(np.max(all_igs)),
            "median": float(np.median(all_igs)),
        }
    
    if all_entropies:
        summary["entropy_statistics"] = {
            "mean": float(np.mean(all_entropies)),
            "std": float(np.std(all_entropies)),
            "min": float(np.min(all_entropies)),
            "max": float(np.max(all_entropies)),
            "median": float(np.median(all_entropies)),
        }
    
    # 按轮次统计
    rounds_ig = {}
    rounds_entropy = {}
    
    for result in results:
        for round_data in result.rounds:
            rid = round_data.round_id
            if rid not in rounds_ig:
                rounds_ig[rid] = []
                rounds_entropy[rid] = []
            rounds_ig[rid].append(round_data.information_gain)
            rounds_entropy[rid].append(round_data.entropy)
    
    summary["per_round_statistics"] = {}
    for rid in sorted(rounds_ig.keys()):
        summary["per_round_statistics"][f"round_{rid}"] = {
            "count": len(rounds_ig[rid]),
            "avg_ig": float(np.mean(rounds_ig[rid])),
            "avg_entropy": float(np.mean(rounds_entropy[rid])),
            "std_ig": float(np.std(rounds_ig[rid])),
            "std_entropy": float(np.std(rounds_entropy[rid])),
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Summary statistics saved to {output_path}")


def main(
    input_jsonl: str,
    output_json: str,
    output_summary: str = None,
    sampling_k: int = SAMPLING_K,
    use_llm_judge: bool = False,
    save_context: bool = False,
    max_questions: int = None,
    verbose: bool = True,
):
    """
    主函数
    
    Args:
        input_jsonl: 输入 JSONL 文件路径
        output_json: 输出 JSON 文件路径
        output_summary: 汇总统计输出路径（可选）
        sampling_k: Monte Carlo 采样次数
        use_llm_judge: 是否使用 LLM 判断答案等价
        save_context: 是否保存完整上下文
        max_questions: 最大处理问题数（用于测试）
        verbose: 是否打印详细信息
    """
    print("=" * 60)
    print("Information Gain Calculator for Multi-round Agent Dialogues")
    print("=" * 60)
    
    # 加载数据
    print(f"\nLoading data from {input_jsonl}...")
    data = load_jsonl(input_jsonl)
    print(f"Loaded {len(data)} questions")
    
    if max_questions:
        data = data[:max_questions]
        print(f"Processing first {max_questions} questions")
    
    # 初始化计算器
    calculator = IGCalculator(sampling_k=sampling_k)
    
    # 处理每个问题
    results = []
    for i, item in enumerate(tqdm(data, desc="Processing questions")):
        if verbose:
            print(f"\n{'='*40}")
            print(f"Question {i+1}/{len(data)}")
        
        try:
            result = calculator.process_single_question(
                item, 
                use_llm_judge=use_llm_judge,
                verbose=verbose
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            continue
    
    # 保存结果
    print("\nSaving results...")
    save_results(results, output_json, save_context=save_context)
    
    # 保存汇总统计
    if output_summary is None:
        output_summary = output_json.replace('.json', '_summary.json')
    save_summary_statistics(results, output_summary)
    
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Information Gain for agent dialogues")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--summary", "-s", default=None, help="Summary statistics output path")
    parser.add_argument("--sampling-k", "-k", type=int, default=20, help="Number of Monte Carlo samples")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM to judge answer equivalence")
    parser.add_argument("--save-context", action="store_true", help="Save full context in output")
    parser.add_argument("--max-questions", "-n", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    main(
        input_jsonl=args.input,
        output_json=args.output,
        output_summary=args.summary,
        sampling_k=args.sampling_k,
        use_llm_judge=args.use_llm_judge,
        save_context=args.save_context,
        max_questions=args.max_questions,
        verbose=not args.quiet,
    )