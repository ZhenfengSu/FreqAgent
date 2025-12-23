"""
Attention Entropy Analysis for Agent Models
分析模型在 Think、Action、Post-Response 阶段的注意力熵和分配
"""

import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TokenSpan:
    """表示一个 token 区间及其类型"""
    start: int
    end: int
    phase: str  # 'think', 'action', 'observation', 'answer', 'other'


@dataclass
class AttentionStats:
    """存储注意力统计结果"""
    entropy_by_layer: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    attention_to_think: List[float] = field(default_factory=list)
    attention_to_observation: List[float] = field(default_factory=list)
    attention_to_other: List[float] = field(default_factory=list)


class AttentionEntropyAnalyzer:
    """注意力熵分析器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        初始化分析器
        
        Args:
            model_path: 模型路径或 HuggingFace 模型名称
            device: 运行设备
            torch_dtype: 模型精度
        """
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            output_attentions=True  # 关键：启用注意力输出
        )
        self.model.eval()
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers
        
        # 统计结果存储
        self.stats = {
            'think': AttentionStats(),
            'action': AttentionStats(),
            'post_response': AttentionStats()
        }
        
    def parse_messages(self, messages: List[Dict]) -> str:
        """
        将 messages 格式转换为模型输入文本
        
        Args:
            messages: 对话消息列表
            
        Returns:
            完整的输入文本
        """
        # 使用 tokenizer 的 chat template（如果有）
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                return text
            except Exception:
                pass
        
        # 手动拼接
        text_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                text_parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                text_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                text_parts.append(f"<|assistant|>\n{content}")
        
        return "\n".join(text_parts)
    
    def identify_token_spans(
        self, 
        text: str, 
        input_ids: torch.Tensor
    ) -> List[TokenSpan]:
        """
        识别文本中不同阶段的 token 区间
        
        Args:
            text: 输入文本
            input_ids: token IDs
            
        Returns:
            TokenSpan 列表
        """
        spans = []
        
        # 定义正则模式
        patterns = {
            'think': r'<think>(.*?)</think>',
            'action': r'<tool_call>(.*?)</tool_call>',
            'observation': r'<tool_response>(.*?)</tool_response>',
            'answer': r'<answer>(.*?)</answer>'
        }
        
        # 找到所有匹配的文本区间
        text_spans = []
        for phase, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                text_spans.append({
                    'start': match.start(),
                    'end': match.end(),
                    'phase': phase,
                    'content': match.group(0)
                })
        
        # 按位置排序
        text_spans.sort(key=lambda x: x['start'])
        
        # 将文本位置映射到 token 位置
        # 使用 offset_mapping 来精确映射
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        offset_mapping = encoding['offset_mapping'][0].tolist()
        
        for text_span in text_spans:
            start_char = text_span['start']
            end_char = text_span['end']
            
            # 找到对应的 token 区间
            start_token = None
            end_token = None
            
            for idx, (char_start, char_end) in enumerate(offset_mapping):
                if char_start <= start_char < char_end and start_token is None:
                    start_token = idx
                if char_start < end_char <= char_end:
                    end_token = idx + 1
                    break
            
            if start_token is not None and end_token is not None:
                spans.append(TokenSpan(
                    start=start_token,
                    end=end_token,
                    phase=text_span['phase']
                ))
        
        return spans
    
    def compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """
        计算注意力分布的熵
        
        Args:
            attention_weights: 注意力权重 [seq_len] or [num_heads, seq_len]
            
        Returns:
            熵值
        """
        if attention_weights.dim() > 1:
            # 如果是多头，取平均
            attention_weights = attention_weights.mean(dim=0)
        
        # 避免 log(0)
        attention_weights = attention_weights.clamp(min=1e-10)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights))
        
        return entropy.item()
    
    def compute_attention_allocation(
        self,
        attention_weights: torch.Tensor,
        token_spans: List[TokenSpan],
        current_pos: int
    ) -> Tuple[float, float, float]:
        """
        计算注意力在不同区域的分配比例
        
        Args:
            attention_weights: 注意力权重 [num_heads, seq_len] or [seq_len]
            token_spans: token 区间列表
            current_pos: 当前 token 位置
            
        Returns:
            (attention_to_think, attention_to_observation, attention_to_other)
        """
        if attention_weights.dim() > 1:
            # 多头平均
            attention_weights = attention_weights.mean(dim=0)
        
        # 只考虑当前位置之前的注意力（因果掩码）
        attention_weights = attention_weights[:current_pos + 1]
        
        think_score = 0.0
        obs_score = 0.0
        other_score = 0.0
        
        for idx in range(len(attention_weights)):
            weight = attention_weights[idx].item()
            
            # 确定这个位置属于哪个区间
            found_phase = None
            for span in token_spans:
                if span.start <= idx < span.end:
                    found_phase = span.phase
                    break
            
            if found_phase == 'think':
                think_score += weight
            elif found_phase == 'observation':
                obs_score += weight
            else:
                other_score += weight
        
        return think_score, obs_score, other_score
    
    def analyze_sample(
        self,
        messages: List[Dict],
        max_length: int = 4096
    ) -> Dict:
        """
        分析单个样本
        
        Args:
            messages: 对话消息列表
            max_length: 最大序列长度
            
        Returns:
            样本级别的统计结果
        """
        # 提取 assistant 消息并拼接
        assistant_contents = []
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_contents.append(msg['content'])
        
        # 构建完整输入
        full_text = self.parse_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        
        # 识别 token 区间
        token_spans = self.identify_token_spans(full_text, input_ids)
        
        if not token_spans:
            return None
        
        # 获取注意力权重
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # Tuple of [batch, num_heads, seq_len, seq_len]
        
        sample_stats = {
            'think': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []},
            'action': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []},
            'post_response': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []}
        }
        
        # 找到 observation 区间的结束位置，用于识别 post-response
        obs_end_positions = []
        for span in token_spans:
            if span.phase == 'observation':
                obs_end_positions.append(span.end)
        
        # 遍历每个区间
        for span in token_spans:
            if span.phase == 'think':
                phase_key = 'think'
            elif span.phase == 'action':
                phase_key = 'action'
            else:
                continue
            
            # 遍历区间内的每个 token
            for pos in range(span.start, min(span.end, seq_len)):
                if pos == 0:
                    continue  # 跳过第一个 token
                
                # 遍历每一层
                for layer_idx in range(self.num_layers):
                    # 获取当前层的注意力: [batch, num_heads, seq_len, seq_len]
                    layer_attn = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
                    
                    # 获取当前位置的注意力分布: [num_heads, seq_len]
                    pos_attn = layer_attn[:, pos, :pos + 1]  # 因果掩码
                    
                    # 计算熵
                    entropy = self.compute_entropy(pos_attn)
                    sample_stats[phase_key]['entropy_by_layer'][layer_idx].append(entropy)
                
                # 计算注意力分配（使用最后一层）
                last_layer_attn = attentions[-1][0][:, pos, :pos + 1]
                think_score, obs_score, _ = self.compute_attention_allocation(
                    last_layer_attn, token_spans, pos
                )
                sample_stats[phase_key]['attn_think'].append(think_score)
                sample_stats[phase_key]['attn_obs'].append(obs_score)
        
        # 分析 Post-Response Phase
        # 找到 observation 结束后的第一个 token
        for obs_end in obs_end_positions:
            if obs_end < seq_len:
                pos = obs_end  # observation 结束后的第一个 token
                
                for layer_idx in range(self.num_layers):
                    layer_attn = attentions[layer_idx][0]
                    pos_attn = layer_attn[:, pos, :pos + 1]
                    entropy = self.compute_entropy(pos_attn)
                    sample_stats['post_response']['entropy_by_layer'][layer_idx].append(entropy)
                
                # 注意力分配
                last_layer_attn = attentions[-1][0][:, pos, :pos + 1]
                think_score, obs_score, _ = self.compute_attention_allocation(
                    last_layer_attn, token_spans, pos
                )
                sample_stats['post_response']['attn_think'].append(think_score)
                sample_stats['post_response']['attn_obs'].append(obs_score)
        
        return sample_stats
    
    def analyze_dataset(
        self,
        json_path: str,
        max_samples: Optional[int] = None
    ):
        """
        分析整个数据集
        
        Args:
            json_path: JSON 文件路径
            max_samples: 最大样本数（用于调试）
        """
        # 加载数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"Analyzing {len(data)} samples...")
        
        # 遍历样本
        for item in tqdm(data):
            messages = item['messages']
            sample_stats = self.analyze_sample(messages)
            
            if sample_stats is None:
                continue
            
            # 累积统计
            for phase in ['think', 'action', 'post_response']:
                for layer_idx, entropies in sample_stats[phase]['entropy_by_layer'].items():
                    self.stats[phase].entropy_by_layer[layer_idx].extend(entropies)
                self.stats[phase].attention_to_think.extend(sample_stats[phase]['attn_think'])
                self.stats[phase].attention_to_observation.extend(sample_stats[phase]['attn_obs'])
        
        print("Analysis complete!")
    
    def get_summary(self) -> Dict:
        """
        获取统计摘要
        
        Returns:
            统计摘要字典
        """
        summary = {}
        
        for phase in ['think', 'action', 'post_response']:
            phase_summary = {
                'avg_entropy_by_layer': {},
                'std_entropy_by_layer': {},
                'avg_attention_to_think': 0.0,
                'avg_attention_to_observation': 0.0,
                'num_samples': 0
            }
            
            # 计算每层的平均熵
            for layer_idx, entropies in self.stats[phase].entropy_by_layer.items():
                if entropies:
                    phase_summary['avg_entropy_by_layer'][layer_idx] = np.mean(entropies)
                    phase_summary['std_entropy_by_layer'][layer_idx] = np.std(entropies)
            
            # 计算注意力分配
            if self.stats[phase].attention_to_think:
                phase_summary['avg_attention_to_think'] = np.mean(self.stats[phase].attention_to_think)
                phase_summary['avg_attention_to_observation'] = np.mean(self.stats[phase].attention_to_observation)
                phase_summary['num_samples'] = len(self.stats[phase].attention_to_think)
            
            summary[phase] = phase_summary
        
        return summary
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        绘制分析结果
        
        Args:
            save_path: 图片保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 各阶段各层的平均熵
        ax1 = axes[0, 0]
        colors = {'think': 'blue', 'action': 'red', 'post_response': 'green'}
        labels = {'think': 'Think Phase', 'action': 'Action Phase', 'post_response': 'Post-Response'}
        
        for phase in ['think', 'action', 'post_response']:
            layers = sorted(self.stats[phase].entropy_by_layer.keys())
            avg_entropies = [np.mean(self.stats[phase].entropy_by_layer[l]) 
                           for l in layers if self.stats[phase].entropy_by_layer[l]]
            if avg_entropies:
                ax1.plot(layers[:len(avg_entropies)], avg_entropies, 
                        color=colors[phase], label=labels[phase], linewidth=2)
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Average Entropy', fontsize=12)
        ax1.set_title('Average Attention Entropy by Layer', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 深层网络（最后5层）的熵对比
        ax2 = axes[0, 1]
        last_layers = list(range(max(0, self.num_layers - 5), self.num_layers))
        
        bar_data = []
        for phase in ['think', 'action', 'post_response']:
            layer_entropies = []
            for l in last_layers:
                if l in self.stats[phase].entropy_by_layer and self.stats[phase].entropy_by_layer[l]:
                    layer_entropies.extend(self.stats[phase].entropy_by_layer[l])
            if layer_entropies:
                bar_data.append(np.mean(layer_entropies))
            else:
                bar_data.append(0)
        
        x_pos = np.arange(3)
        bars = ax2.bar(x_pos, bar_data, color=[colors[p] for p in ['think', 'action', 'post_response']])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Think', 'Action', 'Post-Response'])
        ax2.set_ylabel('Average Entropy', fontsize=12)
        ax2.set_title(f'Deep Layers (Last 5) Average Entropy', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 注意力分配对比 - Think vs Observation
        ax3 = axes[1, 0]
        
        attn_data = {'Think': [], 'Observation': []}
        phases = ['think', 'action', 'post_response']
        
        for phase in phases:
            attn_data['Think'].append(
                np.mean(self.stats[phase].attention_to_think) 
                if self.stats[phase].attention_to_think else 0
            )
            attn_data['Observation'].append(
                np.mean(self.stats[phase].attention_to_observation)
                if self.stats[phase].attention_to_observation else 0
            )
        
        x = np.arange(len(phases))
        width = 0.35
        
        ax3.bar(x - width/2, attn_data['Think'], width, label='Attention to Think', color='steelblue')
        ax3.bar(x + width/2, attn_data['Observation'], width, label='Attention to Observation', color='coral')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Think', 'Action', 'Post-Response'])
        ax3.set_ylabel('Attention Proportion', fontsize=12)
        ax3.set_title('Attention Allocation: Think vs Observation', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 注意力分配的详细分布（Action Phase）
        ax4 = axes[1, 1]
        
        if self.stats['action'].attention_to_think and self.stats['action'].attention_to_observation:
            data_to_plot = [
                self.stats['action'].attention_to_think,
                self.stats['action'].attention_to_observation
            ]
            bp = ax4.boxplot(data_to_plot, labels=['To Think', 'To Observation'])
            ax4.set_ylabel('Attention Score', fontsize=12)
            ax4.set_title('Action Phase: Attention Distribution', fontsize=14)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def save_results(self, save_path: str):
        """
        保存分析结果到 JSON
        
        Args:
            save_path: 保存路径
        """
        summary = self.get_summary()
        
        # 转换为可序列化格式
        serializable_summary = {}
        for phase, stats in summary.items():
            serializable_summary[phase] = {
                'avg_entropy_by_layer': {str(k): v for k, v in stats['avg_entropy_by_layer'].items()},
                'std_entropy_by_layer': {str(k): v for k, v in stats['std_entropy_by_layer'].items()},
                'avg_attention_to_think': float(stats['avg_attention_to_think']),
                'avg_attention_to_observation': float(stats['avg_attention_to_observation']),
                'num_samples': stats['num_samples']
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {save_path}")


def compare_models(
    baseline_path: str,
    ablation_path: str,
    data_path: str,
    output_dir: str = "./results",
    max_samples: Optional[int] = None
):
    """
    对比 Baseline 和 Ablation 模型
    
    Args:
        baseline_path: Baseline 模型路径
        ablation_path: Ablation 模型路径
        data_path: 数据文件路径
        output_dir: 输出目录
        max_samples: 最大样本数
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 分析 Baseline
    print("=" * 50)
    print("Analyzing Baseline Model...")
    print("=" * 50)
    baseline_analyzer = AttentionEntropyAnalyzer(baseline_path)
    baseline_analyzer.analyze_dataset(data_path, max_samples)
    results['baseline'] = baseline_analyzer.get_summary()
    baseline_analyzer.save_results(os.path.join(output_dir, "baseline_results.json"))
    baseline_analyzer.plot_results(os.path.join(output_dir, "baseline_plots.png"))
    
    # 分析 Ablation
    print("\n" + "=" * 50)
    print("Analyzing Ablation Model...")
    print("=" * 50)
    ablation_analyzer = AttentionEntropyAnalyzer(ablation_path)
    ablation_analyzer.analyze_dataset(data_path, max_samples)
    results['ablation'] = ablation_analyzer.get_summary()
    ablation_analyzer.save_results(os.path.join(output_dir, "ablation_results.json"))
    ablation_analyzer.plot_results(os.path.join(output_dir, "ablation_plots.png"))
    
    # 对比分析
    print("\n" + "=" * 50)
    print("Comparison Summary")
    print("=" * 50)
    
    for phase in ['think', 'action', 'post_response']:
        print(f"\n[{phase.upper()} PHASE]")
        
        # 深层熵对比
        baseline_deep_entropy = []
        ablation_deep_entropy = []
        
        num_layers = max(
            max(int(k) for k in results['baseline'][phase]['avg_entropy_by_layer'].keys()) + 1
            if results['baseline'][phase]['avg_entropy_by_layer'] else 0,
            max(int(k) for k in results['ablation'][phase]['avg_entropy_by_layer'].keys()) + 1
            if results['ablation'][phase]['avg_entropy_by_layer'] else 0
        )
        
        for l in range(max(0, num_layers - 5), num_layers):
            if str(l) in results['baseline'][phase]['avg_entropy_by_layer']:
                baseline_deep_entropy.append(results['baseline'][phase]['avg_entropy_by_layer'][str(l)])
            if str(l) in results['ablation'][phase]['avg_entropy_by_layer']:
                ablation_deep_entropy.append(results['ablation'][phase]['avg_entropy_by_layer'][str(l)])
        
        if baseline_deep_entropy and ablation_deep_entropy:
            print(f"  Deep Layer Entropy - Baseline: {np.mean(baseline_deep_entropy):.4f}, "
                  f"Ablation: {np.mean(ablation_deep_entropy):.4f}, "
                  f"Δ: {np.mean(ablation_deep_entropy) - np.mean(baseline_deep_entropy):.4f}")
        
        # 注意力分配对比
        print(f"  Attention to Think - Baseline: {results['baseline'][phase]['avg_attention_to_think']:.4f}, "
              f"Ablation: {results['ablation'][phase]['avg_attention_to_think']:.4f}")
        print(f"  Attention to Observation - Baseline: {results['baseline'][phase]['avg_attention_to_observation']:.4f}, "
              f"Ablation: {results['ablation'][phase]['avg_attention_to_observation']:.4f}")
    
    return results


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Entropy Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_path", type=str, required=True, help="JSON data path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare two models")
    parser.add_argument("--ablation_path", type=str, default=None, help="Ablation model path for comparison")
    
    args = parser.parse_args()
    
    if args.compare and args.ablation_path:
        # 对比两个模型
        compare_models(
            baseline_path=args.model_path,
            ablation_path=args.ablation_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
    else:
        # 分析单个模型
        analyzer = AttentionEntropyAnalyzer(args.model_path)
        analyzer.analyze_dataset(args.data_path, args.max_samples)
        
        # 打印摘要
        summary = analyzer.get_summary()
        print("\n" + "=" * 50)
        print("Analysis Summary")
        print("=" * 50)
        
        for phase, stats in summary.items():
            print(f"\n[{phase.upper()}]")
            print(f"  Samples: {stats['num_samples']}")
            print(f"  Avg Attention to Think: {stats['avg_attention_to_think']:.4f}")
            print(f"  Avg Attention to Observation: {stats['avg_attention_to_observation']:.4f}")
        
        # 保存和绘图
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        analyzer.save_results(os.path.join(args.output_dir, "results.json"))
        analyzer.plot_results(os.path.join(args.output_dir, "plots.png"))