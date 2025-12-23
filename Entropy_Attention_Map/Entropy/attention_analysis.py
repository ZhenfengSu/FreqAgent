"""
Attention Entropy Analysis for Agent Models (Multi-GPU Optimized)
分析模型在 Think、Action、Post-Response 阶段的注意力熵和分配

核心分析策略：
1. Phase A (Think): 生成 <think>...</think> 时的注意力熵
2. Phase B (Action): 生成 <tool_call>...</tool_call> 时的注意力熵
3. Phase C (Post-Response): 收到 <tool_response> 后生成内容时的注意力熵

指标：
- 各阶段各层的平均熵
- 注意力分配比例 (Attention to Think / Observation / Action)
"""

import json
import re
import os
import gc
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TokenSpan:
    """表示一个 token 区间及其类型"""
    start: int
    end: int
    phase: str  # 'think', 'action', 'observation', 'other'


@dataclass
class AttentionStats:
    """存储注意力统计结果"""
    entropy_by_layer: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    attention_to_think: List[float] = field(default_factory=list)
    attention_to_observation: List[float] = field(default_factory=list)
    attention_to_action: List[float] = field(default_factory=list)
    attention_to_other: List[float] = field(default_factory=list)


class AttentionEntropyAnalyzer:
    """注意力熵分析器 (多GPU优化版)"""
    
    def __init__(
        self,
        model_path: str,
        device_map: Union[str, Dict] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_memory: Optional[Dict] = None,
        offload_folder: str = "./offload",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        low_cpu_mem_usage: bool = True,
    ):
        """
        初始化分析器
        
        Args:
            model_path: 模型路径或 HuggingFace 模型名称
            device_map: 设备映射策略，"auto" 自动分配
            torch_dtype: 模型精度
            max_memory: 每个设备的最大内存限制
            offload_folder: CPU/磁盘 offload 文件夹
            load_in_8bit: 使用8bit量化加载
            load_in_4bit: 使用4bit量化加载
            low_cpu_mem_usage: 低CPU内存使用模式
        """
        self.model_path = model_path
        self.offload_folder = offload_folder
        os.makedirs(offload_folder, exist_ok=True)
        
        print(f"Loading model from {model_path}...")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            # 关键修复: 使用 eager attention 以支持 output_attentions
            "attn_implementation": "eager",
        }
        
        # 量化配置
        if load_in_4bit:
            print("Loading model in 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        elif load_in_8bit:
            print("Loading model in 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        
        # 设置内存限制
        if max_memory is None and torch.cuda.device_count() > 0:
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory
                max_memory[i] = f"{int(total_mem * 0.80 / 1024**3)}GB"
            max_memory["cpu"] = "50GB"
            print(f"Auto-configured max_memory: {max_memory}")
        
        # 加载模型
        print("Loading with eager attention (required for output_attentions=True)...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading with eager attention: {e}")
            print("Trying alternative loading method...")
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                **model_kwargs
            )
            if hasattr(self.model.config, '_attn_implementation'):
                self.model.config._attn_implementation = "eager"
        
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        
        if hasattr(self.model, 'hf_device_map'):
            print(f"Model device map: {self._summarize_device_map(self.model.hf_device_map)}")
        
        # 验证注意力输出
        self._verify_attention_output()
        
        # 统计结果存储
        self.stats = {
            'think': AttentionStats(),
            'action': AttentionStats(),
            'post_response': AttentionStats()
        }
    
    def _verify_attention_output(self):
        """验证模型是否支持输出注意力权重"""
        print("Verifying attention output support...")
        try:
            test_input = self.tokenizer("test", return_tensors="pt")
            device = next(self.model.parameters()).device
            test_input = {k: v.to(device) for k, v in test_input.items()}
            
            with torch.no_grad():
                outputs = self.model(**test_input, output_attentions=True, use_cache=False)
            
            if outputs.attentions is None:
                raise ValueError("Model returned None for attentions")
            
            print(f"✓ Attention output verified. Got {len(outputs.attentions)} layers.")
            print(f"  Attention shape per layer: {outputs.attentions[0].shape}")
            
            del outputs, test_input
            self.clear_memory()
            
        except Exception as e:
            print(f"✗ Attention output verification failed: {e}")
            raise RuntimeError(
                "Model does not support output_attentions=True. "
                "Please ensure the model is loaded with attn_implementation='eager'"
            )
    
    def _summarize_device_map(self, device_map: Dict) -> Dict:
        """汇总设备映射"""
        summary = defaultdict(list)
        for layer, device in device_map.items():
            summary[str(device)].append(layer)
        return {k: f"{len(v)} layers" for k, v in summary.items()}
    
    def clear_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def extract_assistant_content(self, messages: List[Dict]) -> str:
        """
        只提取 assistant 角色的内容，拼接成完整文本
        """
        assistant_parts = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if content:
                    assistant_parts.append(content)
        
        return "\n".join(assistant_parts)
    
    def parse_messages_for_context(self, messages: List[Dict]) -> str:
        """
        将完整 messages 转换为模型输入文本（用于获取完整上下文）
        """
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
        
        # 备用方案
        text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
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
        
        Returns:
            List of TokenSpan with phases: 'think', 'action', 'observation', 'other'
        """
        spans = []
        
        # 定义各阶段的正则模式
        patterns = {
            'think': r'<think>(.*?)</think>',
            'action': r'<tool_call>(.*?)</tool_call>',
            'observation': r'<tool_response>(.*?)</tool_response>',
        }
        
        # 首先在文本中找到所有匹配
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
        
        # 将字符位置映射到 token 位置
        try:
            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors='pt',
                add_special_tokens=False
            )
            offset_mapping = encoding['offset_mapping'][0].tolist()
        except Exception:
            return spans
        
        for text_span in text_spans:
            start_char = text_span['start']
            end_char = text_span['end']
            
            start_token = None
            end_token = None
            
            for idx, (char_start, char_end) in enumerate(offset_mapping):
                if char_end == 0:  # 特殊token
                    continue
                if start_token is None and char_end > start_char:
                    start_token = idx
                if char_start < end_char:
                    end_token = idx + 1
            
            if start_token is not None and end_token is not None:
                spans.append(TokenSpan(
                    start=start_token,
                    end=min(end_token, len(offset_mapping)),
                    phase=text_span['phase']
                ))
        
        return spans
    
    def compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """
        计算注意力分布的熵
        
        Args:
            attention_weights: [num_heads, seq_len] 或 [seq_len]
            
        Returns:
            平均熵值
        """
        # 如果是多头，先平均
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=0)
        
        # 确保数值稳定
        attention_weights = attention_weights.float().clamp(min=1e-10)
        
        # 归一化（确保和为1）
        attention_weights = attention_weights / attention_weights.sum()
        
        # 计算熵: H = -sum(p * log(p))
        entropy = -torch.sum(attention_weights * torch.log(attention_weights))
        
        return entropy.item()
    
    def compute_attention_allocation(
        self,
        attention_weights: torch.Tensor,
        token_spans: List[TokenSpan],
        current_pos: int
    ) -> Dict[str, float]:
        """
        计算注意力在不同区域的分配比例（除以长度归一化）
        
        Args:
            attention_weights: [num_heads, seq_len] 或 [seq_len]
            token_spans: token 区间列表
            current_pos: 当前位置
            
        Returns:
            各区域的归一化注意力分数
        """
        # 如果是多头，平均
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=0)
        
        # 只看当前位置之前的注意力
        attention_weights = attention_weights[:current_pos + 1].float()
        
        # 统计各区域的总注意力和长度
        region_scores = {'think': 0.0, 'observation': 0.0, 'action': 0.0, 'other': 0.0}
        region_lengths = {'think': 0, 'observation': 0, 'action': 0, 'other': 0}
        
        # 为每个位置标记所属区域
        position_to_phase = {}
        for span in token_spans:
            if span.end <= current_pos + 1:  # 只考虑当前位置之前的span
                for idx in range(span.start, min(span.end, current_pos + 1)):
                    position_to_phase[idx] = span.phase
        
        # 累计各区域的注意力
        for idx in range(len(attention_weights)):
            weight = attention_weights[idx].item()
            phase = position_to_phase.get(idx, 'other')
            region_scores[phase] += weight
            region_lengths[phase] += 1
        
        # 归一化：除以长度
        normalized_scores = {}
        for phase in region_scores:
            if region_lengths[phase] > 0:
                normalized_scores[phase] = region_scores[phase] / region_lengths[phase]
            else:
                normalized_scores[phase] = 0.0
        
        return normalized_scores
    
    def analyze_sample(
        self,
        messages: List[Dict],
        max_length: int = 4096,
        position_sample_rate: float = 0.3
    ) -> Optional[Dict]:
        """
        分析单个样本
        
        Args:
            messages: 对话消息列表
            max_length: 最大序列长度
            position_sample_rate: 位置采样率
            
        Returns:
            样本级别的统计结果
        """
        # 获取完整文本用于分析
        full_text = self.parse_messages_for_context(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        
        # 识别各阶段的 token 区间
        token_spans = self.identify_token_spans(full_text, input_ids)
        
        if not token_spans:
            return None
        
        # 初始化样本统计
        sample_stats = {
            'think': {'entropy_by_layer': defaultdict(list), 
                     'attn_think': [], 'attn_obs': [], 'attn_action': []},
            'action': {'entropy_by_layer': defaultdict(list), 
                      'attn_think': [], 'attn_obs': [], 'attn_action': []},
            'post_response': {'entropy_by_layer': defaultdict(list), 
                             'attn_think': [], 'attn_obs': [], 'attn_action': []}
        }
        
        # 找到需要分析的位置
        positions_by_phase = {'think': [], 'action': [], 'post_response': []}
        
        # Phase A & B: Think 和 Action 阶段
        for span in token_spans:
            if span.phase == 'think':
                # 生成 think 内容时的位置（不包含开始标签，模拟生成过程）
                phase_positions = list(range(span.start + 1, min(span.end, seq_len)))
                positions_by_phase['think'].extend(phase_positions)
            elif span.phase == 'action':
                # 生成 action 内容时的位置
                phase_positions = list(range(span.start + 1, min(span.end, seq_len)))
                positions_by_phase['action'].extend(phase_positions)
        
        # Phase C: Post-Response（收到 tool_response 后的下一个 token）
        for i, span in enumerate(token_spans):
            if span.phase == 'observation':
                # 找到 observation 结束后的位置
                post_obs_start = span.end
                # 找到下一个 span 的开始或序列结束
                next_span_start = seq_len
                for next_span in token_spans:
                    if next_span.start > span.end:
                        next_span_start = next_span.start
                        break
                
                # Post-response 位置
                if post_obs_start < seq_len:
                    post_positions = list(range(post_obs_start, min(post_obs_start + 20, next_span_start, seq_len)))
                    positions_by_phase['post_response'].extend(post_positions)
        
        # 采样位置以减少计算量
        for phase in positions_by_phase:
            positions = positions_by_phase[phase]
            if position_sample_rate < 1.0 and len(positions) > 10:
                num_samples = max(5, int(len(positions) * position_sample_rate))
                indices = np.linspace(0, len(positions) - 1, num_samples, dtype=int)
                positions_by_phase[phase] = [positions[i] for i in indices]
        
        # 检查是否有足够的位置
        total_positions = sum(len(p) for p in positions_by_phase.values())
        if total_positions == 0:
            return None
        
        try:
            with torch.no_grad():
                # 前向传播获取注意力
                outputs = self.model(
                    **inputs, 
                    output_attentions=True,
                    use_cache=False
                )
                attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]
                
                # 处理每个阶段的位置
                for phase, positions in positions_by_phase.items():
                    for pos in positions:
                        if pos >= seq_len:
                            continue
                            
                        # 遍历每一层计算熵
                        for layer_idx in range(self.num_layers):
                            # [batch, num_heads, seq_len, seq_len] -> [num_heads, pos+1]
                            # 关键：只看位置 pos 对之前所有位置的注意力（模拟生成）
                            layer_attn = attentions[layer_idx][0]  # [heads, seq, seq]
                            pos_attn = layer_attn[:, pos, :pos + 1].cpu()  # [heads, pos+1]
                            
                            entropy = self.compute_entropy(pos_attn)
                            sample_stats[phase]['entropy_by_layer'][layer_idx].append(entropy)
                        
                        # 计算注意力分配（使用最后一层，更能反映最终决策）
                        last_layer_attn = attentions[-1][0][:, pos, :pos + 1].cpu()
                        allocation = self.compute_attention_allocation(
                            last_layer_attn, token_spans, pos
                        )
                        sample_stats[phase]['attn_think'].append(allocation['think'])
                        sample_stats[phase]['attn_obs'].append(allocation['observation'])
                        sample_stats[phase]['attn_action'].append(allocation['action'])
                
                del outputs, attentions
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM for sequence length {seq_len}, skipping...")
                self.clear_memory()
                return None
            raise
        
        return sample_stats
    
    def analyze_dataset(
        self,
        json_path: str,
        max_samples: Optional[int] = None,
        save_interval: int = 50,
        checkpoint_dir: Optional[str] = None
    ):
        """
        分析整个数据集
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"Analyzing {len(data)} samples...")
        
        # 检查检查点
        start_idx = 0
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                start_idx = checkpoint.get('last_idx', 0) + 1
                self._load_stats(checkpoint.get('stats', {}))
                print(f"Resuming from index {start_idx}")
        
        success_count = 0
        error_count = 0
        
        for idx in tqdm(range(start_idx, len(data)), initial=start_idx, total=len(data)):
            item = data[idx]
            messages = item.get('messages', item.get('conversations', []))
            
            try:
                sample_stats = self.analyze_sample(messages)
                
                if sample_stats is not None:
                    # 累积统计
                    for phase in ['think', 'action', 'post_response']:
                        for layer_idx, entropies in sample_stats[phase]['entropy_by_layer'].items():
                            self.stats[phase].entropy_by_layer[layer_idx].extend(entropies)
                        self.stats[phase].attention_to_think.extend(sample_stats[phase]['attn_think'])
                        self.stats[phase].attention_to_observation.extend(sample_stats[phase]['attn_obs'])
                        self.stats[phase].attention_to_action.extend(sample_stats[phase]['attn_action'])
                    success_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Error processing sample {idx}: {e}")
                continue
            
            # 定期清理内存
            if (idx + 1) % 5 == 0:
                self.clear_memory()
            
            # 保存检查点
            if checkpoint_dir and (idx + 1) % save_interval == 0:
                self._save_checkpoint(checkpoint_dir, idx)
        
        print(f"Analysis complete! Success: {success_count}, Errors: {error_count}")
    
    def _save_checkpoint(self, checkpoint_dir: str, last_idx: int):
        """保存检查点"""
        checkpoint = {
            'last_idx': last_idx,
            'stats': self._serialize_stats()
        }
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def _serialize_stats(self) -> Dict:
        """序列化统计数据"""
        serialized = {}
        for phase in ['think', 'action', 'post_response']:
            serialized[phase] = {
                'entropy_by_layer': {
                    str(k): v for k, v in self.stats[phase].entropy_by_layer.items()
                },
                'attention_to_think': self.stats[phase].attention_to_think,
                'attention_to_observation': self.stats[phase].attention_to_observation,
                'attention_to_action': self.stats[phase].attention_to_action,
            }
        return serialized
    
    def _load_stats(self, serialized: Dict):
        """加载统计数据"""
        for phase in ['think', 'action', 'post_response']:
            if phase in serialized:
                for k, v in serialized[phase].get('entropy_by_layer', {}).items():
                    self.stats[phase].entropy_by_layer[int(k)] = v
                self.stats[phase].attention_to_think = serialized[phase].get('attention_to_think', [])
                self.stats[phase].attention_to_observation = serialized[phase].get('attention_to_observation', [])
                self.stats[phase].attention_to_action = serialized[phase].get('attention_to_action', [])
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        summary = {}
        
        for phase in ['think', 'action', 'post_response']:
            phase_summary = {
                'avg_entropy_by_layer': {},
                'std_entropy_by_layer': {},
                'avg_attention_to_think': 0.0,
                'avg_attention_to_observation': 0.0,
                'avg_attention_to_action': 0.0,
                'std_attention_to_think': 0.0,
                'std_attention_to_observation': 0.0,
                'std_attention_to_action': 0.0,
                'num_positions': 0
            }
            
            # 各层熵统计
            for layer_idx, entropies in self.stats[phase].entropy_by_layer.items():
                if entropies:
                    phase_summary['avg_entropy_by_layer'][layer_idx] = float(np.mean(entropies))
                    phase_summary['std_entropy_by_layer'][layer_idx] = float(np.std(entropies))
            
            # 注意力分配统计
            if self.stats[phase].attention_to_think:
                phase_summary['avg_attention_to_think'] = float(np.mean(self.stats[phase].attention_to_think))
                phase_summary['std_attention_to_think'] = float(np.std(self.stats[phase].attention_to_think))
            if self.stats[phase].attention_to_observation:
                phase_summary['avg_attention_to_observation'] = float(np.mean(self.stats[phase].attention_to_observation))
                phase_summary['std_attention_to_observation'] = float(np.std(self.stats[phase].attention_to_observation))
            if self.stats[phase].attention_to_action:
                phase_summary['avg_attention_to_action'] = float(np.mean(self.stats[phase].attention_to_action))
                phase_summary['std_attention_to_action'] = float(np.std(self.stats[phase].attention_to_action))
            
            phase_summary['num_positions'] = len(self.stats[phase].attention_to_think)
            
            summary[phase] = phase_summary
        
        return summary
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制分析结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        colors = {'think': '#2ecc71', 'action': '#e74c3c', 'post_response': '#3498db'}
        labels = {'think': 'Think Phase', 'action': 'Action Phase', 'post_response': 'Post-Response'}
        
        # 1. 各阶段各层的平均熵
        ax1 = axes[0, 0]
        for phase in ['think', 'action', 'post_response']:
            layers = sorted(self.stats[phase].entropy_by_layer.keys())
            if layers:
                avg_entropies = [np.mean(self.stats[phase].entropy_by_layer[l]) 
                               for l in layers if self.stats[phase].entropy_by_layer[l]]
                if avg_entropies:
                    ax1.plot(layers[:len(avg_entropies)], avg_entropies, 
                            color=colors[phase], label=labels[phase], linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Average Entropy', fontsize=12)
        ax1.set_title('Attention Entropy by Layer', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 深层网络的熵对比
        ax2 = axes[0, 1]
        phases = ['think', 'action', 'post_response']
        bar_data = []
        for phase in phases:
            layers = sorted(self.stats[phase].entropy_by_layer.keys())
            if layers:
                last_layers = layers[-min(5, len(layers)):]
                layer_entropies = []
                for l in last_layers:
                    if self.stats[phase].entropy_by_layer[l]:
                        layer_entropies.extend(self.stats[phase].entropy_by_layer[l])
                bar_data.append(np.mean(layer_entropies) if layer_entropies else 0)
            else:
                bar_data.append(0)
        
        x_pos = np.arange(3)
        bars = ax2.bar(x_pos, bar_data, color=[colors[p] for p in phases])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Think', 'Action', 'Post-Resp'])
        ax2.set_ylabel('Average Entropy', fontsize=12)
        ax2.set_title('Deep Layers (Last 5) Entropy', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, bar_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 注意力分配：Attention to Think
        ax3 = axes[0, 2]
        attn_think = [np.mean(self.stats[p].attention_to_think) if self.stats[p].attention_to_think else 0 
                     for p in phases]
        attn_think_std = [np.std(self.stats[p].attention_to_think) if self.stats[p].attention_to_think else 0 
                        for p in phases]
        
        bars = ax3.bar(x_pos, attn_think, yerr=attn_think_std, color=[colors[p] for p in phases], 
                      capsize=5, alpha=0.8)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['Think', 'Action', 'Post-Resp'])
        ax3.set_ylabel('Normalized Attention Score', fontsize=12)
        ax3.set_title('Attention to Think Region', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 注意力分配：Attention to Observation
        ax4 = axes[1, 0]
        attn_obs = [np.mean(self.stats[p].attention_to_observation) if self.stats[p].attention_to_observation else 0 
                   for p in phases]
        attn_obs_std = [np.std(self.stats[p].attention_to_observation) if self.stats[p].attention_to_observation else 0 
                       for p in phases]
        
        bars = ax4.bar(x_pos, attn_obs, yerr=attn_obs_std, color=[colors[p] for p in phases],
                      capsize=5, alpha=0.8)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['Think', 'Action', 'Post-Resp'])
        ax4.set_ylabel('Normalized Attention Score', fontsize=12)
        ax4.set_title('Attention to Observation (Tool Response)', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 注意力分配对比（堆叠柱状图）
        ax5 = axes[1, 1]
        attn_action = [np.mean(self.stats[p].attention_to_action) if self.stats[p].attention_to_action else 0 
                      for p in phases]
        
        width = 0.25
        x = np.arange(len(phases))
        
        ax5.bar(x - width, attn_think, width, label='To Think', color='#2ecc71', alpha=0.8)
        ax5.bar(x, attn_obs, width, label='To Observation', color='#e74c3c', alpha=0.8)
        ax5.bar(x + width, attn_action, width, label='To Action', color='#3498db', alpha=0.8)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Think', 'Action', 'Post-Resp'])
        ax5.set_ylabel('Normalized Attention Score', fontsize=12)
        ax5.set_title('Attention Allocation Comparison', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 样本数量统计
        ax6 = axes[1, 2]
        sample_counts = [len(self.stats[p].attention_to_think) for p in phases]
        bars = ax6.bar(x_pos, sample_counts, color=[colors[p] for p in phases])
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(['Think', 'Action', 'Post-Resp'])
        ax6.set_ylabel('Number of Positions', fontsize=12)
        ax6.set_title('Analysis Coverage', fontsize=14)
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, sample_counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def save_results(self, save_path: str):
        """保存分析结果到 JSON"""
        summary = self.get_summary()
        
        # 转换为可序列化格式
        serializable_summary = {}
        for phase, stats in summary.items():
            serializable_summary[phase] = {
                'avg_entropy_by_layer': {str(k): v for k, v in stats['avg_entropy_by_layer'].items()},
                'std_entropy_by_layer': {str(k): v for k, v in stats['std_entropy_by_layer'].items()},
                'avg_attention_to_think': stats['avg_attention_to_think'],
                'avg_attention_to_observation': stats['avg_attention_to_observation'],
                'avg_attention_to_action': stats['avg_attention_to_action'],
                'std_attention_to_think': stats['std_attention_to_think'],
                'std_attention_to_observation': stats['std_attention_to_observation'],
                'std_attention_to_action': stats['std_attention_to_action'],
                'num_positions': stats['num_positions']
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {save_path}")


def compare_models(
    baseline_path: str,
    ablation_path: str,
    data_path: str,
    output_dir: str = "./results",
    max_samples: Optional[int] = None,
    device_map: str = "auto",
    max_memory: Optional[Dict] = None,
    load_in_4bit: bool = False
):
    """
    对比 Baseline 和 Ablation 模型
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 分析 Baseline
    print("=" * 70)
    print("Analyzing Baseline Model...")
    print("=" * 70)
    baseline_analyzer = AttentionEntropyAnalyzer(
        baseline_path, 
        device_map=device_map,
        max_memory=max_memory,
        load_in_4bit=load_in_4bit
    )
    baseline_analyzer.analyze_dataset(
        data_path, 
        max_samples,
        checkpoint_dir=os.path.join(output_dir, "baseline_checkpoint")
    )
    results['baseline'] = baseline_analyzer.get_summary()
    baseline_analyzer.save_results(os.path.join(output_dir, "baseline_results.json"))
    baseline_analyzer.plot_results(os.path.join(output_dir, "baseline_plots.png"))
    
    # 清理
    del baseline_analyzer
    gc.collect()
    torch.cuda.empty_cache()
    
    # 分析 Ablation
    print("\n" + "=" * 70)
    print("Analyzing Ablation Model...")
    print("=" * 70)
    ablation_analyzer = AttentionEntropyAnalyzer(
        ablation_path,
        device_map=device_map,
        max_memory=max_memory,
        load_in_4bit=load_in_4bit
    )
    ablation_analyzer.analyze_dataset(
        data_path, 
        max_samples,
        checkpoint_dir=os.path.join(output_dir, "ablation_checkpoint")
    )
    results['ablation'] = ablation_analyzer.get_summary()
    ablation_analyzer.save_results(os.path.join(output_dir, "ablation_results.json"))
    ablation_analyzer.plot_results(os.path.join(output_dir, "ablation_plots.png"))
    
    # 对比分析
    plot_comparison(results, os.path.join(output_dir, "comparison_plots.png"))
    
    # 打印详细对比
    print_comparison_summary(results)
    
    # 保存对比结果
    with open(os.path.join(output_dir, "comparison_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def plot_comparison(results: Dict, save_path: str):
    """绘制模型对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {'baseline': '#3498db', 'ablation': '#e74c3c'}
    phases = ['think', 'action', 'post_response']
    phase_labels = ['Think', 'Action', 'Post-Response']
    
    # 1. Action Phase 熵对比（核心指标）
    ax1 = axes[0, 0]
    for model_type in ['baseline', 'ablation']:
        if 'action' in results[model_type]:
            layers_dict = results[model_type]['action']['avg_entropy_by_layer']
            if layers_dict:
                layers = sorted([int(k) for k in layers_dict.keys()])
                entropies = [layers_dict[str(l)] for l in layers]
                ax1.plot(layers, entropies, color=colors[model_type], 
                        label=model_type.capitalize(), linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Average Entropy', fontsize=12)
    ax1.set_title('Action Phase: Entropy by Layer', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 深层熵对比
    ax2 = axes[0, 1]
    x = np.arange(len(phases))
    width = 0.35
    
    baseline_deep = []
    ablation_deep = []
    
    for phase in phases:
        for model_type, target_list in [('baseline', baseline_deep), ('ablation', ablation_deep)]:
            layers_dict = results[model_type][phase]['avg_entropy_by_layer']
            if layers_dict:
                layers = sorted([int(k) for k in layers_dict.keys()])
                last_layers = layers[-min(5, len(layers)):]
                avg = np.mean([layers_dict[str(l)] for l in last_layers])
                target_list.append(avg)
            else:
                target_list.append(0)
    
    ax2.bar(x - width/2, baseline_deep, width, label='Baseline', color=colors['baseline'])
    ax2.bar(x + width/2, ablation_deep, width, label='Ablation', color=colors['ablation'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(phase_labels)
    ax2.set_ylabel('Deep Layer Entropy', fontsize=12)
    ax2.set_title('Deep Layer (Last 5) Entropy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Attention to Think 对比
    ax3 = axes[0, 2]
    baseline_attn = [results['baseline'][p]['avg_attention_to_think'] for p in phases]
    ablation_attn = [results['ablation'][p]['avg_attention_to_think'] for p in phases]
    
    ax3.bar(x - width/2, baseline_attn, width, label='Baseline', color=colors['baseline'])
    ax3.bar(x + width/2, ablation_attn, width, label='Ablation', color=colors['ablation'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(phase_labels)
    ax3.set_ylabel('Normalized Attention', fontsize=12)
    ax3.set_title('Attention to Think Region', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Attention to Observation 对比
    ax4 = axes[1, 0]
    baseline_obs = [results['baseline'][p]['avg_attention_to_observation'] for p in phases]
    ablation_obs = [results['ablation'][p]['avg_attention_to_observation'] for p in phases]
    
    ax4.bar(x - width/2, baseline_obs, width, label='Baseline', color=colors['baseline'])
    ax4.bar(x + width/2, ablation_obs, width, label='Ablation', color=colors['ablation'])
    ax4.set_xticks(x)
    ax4.set_xticklabels(phase_labels)
    ax4.set_ylabel('Normalized Attention', fontsize=12)
    ax4.set_title('Attention to Observation', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 熵变化量 (Ablation - Baseline)
    ax5 = axes[1, 1]
    entropy_diff = [ablation_deep[i] - baseline_deep[i] for i in range(len(phases))]
    colors_diff = ['#27ae60' if d > 0 else '#c0392b' for d in entropy_diff]
    
    bars = ax5.bar(x, entropy_diff, color=colors_diff)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xticks(x)
    ax5.set_xticklabels(phase_labels)
    ax5.set_ylabel('Entropy Difference (Ablation - Baseline)', fontsize=12)
    ax5.set_title('Entropy Change After Ablation', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, entropy_diff):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        ax5.text(bar.get_x() + bar.get_width()/2, ypos, 
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
    
    # 6. Think/Observation 注意力比例变化
    ax6 = axes[1, 2]
    
    # 计算 Think/Obs 比例
    baseline_ratio = []
    ablation_ratio = []
    for i, p in enumerate(phases):
        b_think = results['baseline'][p]['avg_attention_to_think']
        b_obs = results['baseline'][p]['avg_attention_to_observation']
        a_think = results['ablation'][p]['avg_attention_to_think']
        a_obs = results['ablation'][p]['avg_attention_to_observation']
        
        baseline_ratio.append(b_think / (b_obs + 1e-8))
        ablation_ratio.append(a_think / (a_obs + 1e-8))
    
    ax6.bar(x - width/2, baseline_ratio, width, label='Baseline', color=colors['baseline'])
    ax6.bar(x + width/2, ablation_ratio, width, label='Ablation', color=colors['ablation'])
    ax6.set_xticks(x)
    ax6.set_xticklabels(phase_labels)
    ax6.set_ylabel('Think/Observation Ratio', fontsize=12)
    ax6.set_title('Attention Ratio: Think vs Observation', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def print_comparison_summary(results: Dict):
    """打印详细对比摘要"""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    phases = ['think', 'action', 'post_response']
    phase_names = {'think': 'THINK', 'action': 'ACTION', 'post_response': 'POST-RESPONSE'}
    
    for phase in phases:
        print(f"\n{'─' * 70}")
        print(f"  {phase_names[phase]} PHASE")
        print(f"{'─' * 70}")
        
        b = results['baseline'][phase]
        a = results['ablation'][phase]
        
        # 深层熵
        b_layers = sorted([int(k) for k in b['avg_entropy_by_layer'].keys()])
        a_layers = sorted([int(k) for k in a['avg_entropy_by_layer'].keys()])
        
        if b_layers and a_layers:
            b_deep = np.mean([b['avg_entropy_by_layer'][str(l)] for l in b_layers[-5:]])
            a_deep = np.mean([a['avg_entropy_by_layer'][str(l)] for l in a_layers[-5:]])
            diff_deep = a_deep - b_deep
            
            print(f"\n  Deep Layer Entropy (Last 5 layers):")
            print(f"    Baseline:  {b_deep:.4f}")
            print(f"    Ablation:  {a_deep:.4f}")
            print(f"    Change:    {diff_deep:+.4f} ({'↑' if diff_deep > 0 else '↓'})")
        
        # 注意力分配
        print(f"\n  Attention Allocation (Normalized):")
        
        for target, target_name in [('think', 'Think'), ('observation', 'Observation'), ('action', 'Action')]:
            b_val = b[f'avg_attention_to_{target}']
            a_val = a[f'avg_attention_to_{target}']
            diff = a_val - b_val
            
            print(f"\n    To {target_name}:")
            print(f"      Baseline:  {b_val:.4f}")
            print(f"      Ablation:  {a_val:.4f}")
            print(f"      Change:    {diff:+.4f} ({'↑' if diff > 0 else '↓'})")
        
        # 位置数量
        print(f"\n  Positions Analyzed:")
        print(f"    Baseline:  {b['num_positions']}")
        print(f"    Ablation:  {a['num_positions']}")
    
    # 关键发现
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Action Phase 熵变化
    b_action_layers = sorted([int(k) for k in results['baseline']['action']['avg_entropy_by_layer'].keys()])
    a_action_layers = sorted([int(k) for k in results['ablation']['action']['avg_entropy_by_layer'].keys()])
    
    if b_action_layers and a_action_layers:
        b_action_deep = np.mean([results['baseline']['action']['avg_entropy_by_layer'][str(l)] 
                                 for l in b_action_layers[-5:]])
        a_action_deep = np.mean([results['ablation']['action']['avg_entropy_by_layer'][str(l)] 
                                 for l in a_action_layers[-5:]])
        
        if a_action_deep < b_action_deep:
            print("\n  ✓ ACTION PHASE entropy DECREASED after ablation")
            print(f"    → Supports hypothesis: Model shows 'reflex-like' behavior")
            print(f"    → Entropy drop: {b_action_deep:.4f} → {a_action_deep:.4f} ({a_action_deep - b_action_deep:+.4f})")
        else:
            print("\n  ✗ ACTION PHASE entropy did NOT decrease as expected")
    
    # Think 注意力变化
    b_think_attn = results['baseline']['action']['avg_attention_to_think']
    a_think_attn = results['ablation']['action']['avg_attention_to_think']
    
    if a_think_attn < b_think_attn:
        print("\n  ✓ Attention to THINK region DECREASED in Action Phase")
        print(f"    → Model relies less on thinking when generating actions")
        print(f"    → Attention drop: {b_think_attn:.4f} → {a_think_attn:.4f} ({a_think_attn - b_think_attn:+.4f})")
    
    # Observation 注意力变化
    b_obs_attn = results['baseline']['action']['avg_attention_to_observation']
    a_obs_attn = results['ablation']['action']['avg_attention_to_observation']
    
    if a_obs_attn > b_obs_attn:
        print("\n  ✓ Attention to OBSERVATION increased in Action Phase")
        print(f"    → Model shows direct observation-to-action mapping")
        print(f"    → Attention increase: {b_obs_attn:.4f} → {a_obs_attn:.4f} ({a_obs_attn - b_obs_attn:+.4f})")
    
    print("\n" + "=" * 70)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Entropy Analysis for Agent Models")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_path", type=str, required=True, help="JSON data path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare two models")
    parser.add_argument("--ablation_path", type=str, default=None, help="Ablation model path (for comparison)")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping strategy")
    parser.add_argument("--max_memory_per_gpu", type=str, default=None, help="Max memory per GPU, e.g., '20GB'")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    
    args = parser.parse_args()
    
    # 配置显存限制
    max_memory = None
    if args.max_memory_per_gpu:
        num_gpus = torch.cuda.device_count()
        max_memory = {i: args.max_memory_per_gpu for i in range(num_gpus)}
        max_memory["cpu"] = "50GB"
    
    if args.compare and args.ablation_path:
        # 对比两个模型
        compare_models(
            baseline_path=args.model_path,
            ablation_path=args.ablation_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            device_map=args.device_map,
            max_memory=max_memory,
            load_in_4bit=args.load_in_4bit
        )
    else:
        # 单模型分析
        analyzer = AttentionEntropyAnalyzer(
            args.model_path,
            device_map=args.device_map,
            max_memory=max_memory,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )
        analyzer.analyze_dataset(
            args.data_path, 
            args.max_samples,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoint")
        )
        
        # 打印摘要
        summary = analyzer.get_summary()
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        for phase, stats in summary.items():
            print(f"\n[{phase.upper()}]")
            print(f"  Positions Analyzed: {stats['num_positions']}")
            print(f"  Avg Attention to Think: {stats['avg_attention_to_think']:.4f} ± {stats['std_attention_to_think']:.4f}")
            print(f"  Avg Attention to Observation: {stats['avg_attention_to_observation']:.4f} ± {stats['std_attention_to_observation']:.4f}")
            print(f"  Avg Attention to Action: {stats['avg_attention_to_action']:.4f} ± {stats['std_attention_to_action']:.4f}")
            
            if stats['avg_entropy_by_layer']:
                layers = sorted([int(k) for k in stats['avg_entropy_by_layer'].keys()])
                last_5_entropy = np.mean([stats['avg_entropy_by_layer'][l] for l in layers[-5:]])
                print(f"  Deep Layer Entropy (Last 5): {last_5_entropy:.4f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        analyzer.save_results(os.path.join(args.output_dir, "results.json"))
        analyzer.plot_results(os.path.join(args.output_dir, "plots.png"))