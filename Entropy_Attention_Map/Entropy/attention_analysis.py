"""
Attention Entropy Analysis for Agent Models (Multi-GPU Optimized)
分析模型在 Think、Action、Post-Response 阶段的注意力熵和分配
支持多GPU并行处理和内存优化

修复: SDPA attention 不支持 output_attentions 的问题
"""

import json
import re
import os
import gc
import torch
from torch.utils.data import Dataset
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
    phase: str  # 'think', 'action', 'observation', 'answer', 'other'


@dataclass
class AttentionStats:
    """存储注意力统计结果"""
    entropy_by_layer: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    attention_to_think: List[float] = field(default_factory=list)
    attention_to_observation: List[float] = field(default_factory=list)
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
            device_map: 设备映射策略，"auto" 自动分配，"balanced" 均衡分配
            torch_dtype: 模型精度
            max_memory: 每个设备的最大内存限制，如 {0: "20GB", 1: "20GB", "cpu": "50GB"}
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
                # 使用80%的显存，留更多余量给注意力矩阵
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
            # 备选方案：不指定 attn_implementation
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                **model_kwargs
            )
            # 尝试修改模型配置
            if hasattr(self.model.config, '_attn_implementation'):
                self.model.config._attn_implementation = "eager"
        
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        
        # 打印模型分布信息
        if hasattr(self.model, 'hf_device_map'):
            print(f"Model device map: {self._summarize_device_map(self.model.hf_device_map)}")
        
        # 验证注意力输出是否可用
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
            print("  The analysis may not work correctly.")
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
    
    def parse_messages(self, messages: List[Dict]) -> str:
        """将 messages 格式转换为模型输入文本"""
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
        """识别文本中不同阶段的 token 区间"""
        spans = []
        
        patterns = {
            'think': r'<think>(.*?)</think>',
            'action': r'<tool_call>(.*?)</tool_call>',
            'observation': r'<tool_response>(.*?)</tool_response>',
            'answer': r'<answer>(.*?)</answer>'
        }
        
        text_spans = []
        for phase, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                text_spans.append({
                    'start': match.start(),
                    'end': match.end(),
                    'phase': phase,
                    'content': match.group(0)
                })
        
        text_spans.sort(key=lambda x: x['start'])
        
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
        """计算注意力分布的熵"""
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.float().clamp(min=1e-10)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights))
        
        return entropy.item()
    
    def compute_attention_allocation(
        self,
        attention_weights: torch.Tensor,
        token_spans: List[TokenSpan],
        current_pos: int
    ) -> Tuple[float, float, float]:
        """计算注意力在不同区域的分配比例"""
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights[:current_pos + 1].float()
        
        think_score = 0.0
        obs_score = 0.0
        other_score = 0.0
        
        for idx in range(len(attention_weights)):
            weight = attention_weights[idx].item()
            
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
    
    def analyze_sample_layerwise(
        self,
        messages: List[Dict],
        max_length: int = 4096,
        layers_per_batch: int = 4,
        position_sample_rate: float = 0.3
    ) -> Optional[Dict]:
        """
        逐层分析单个样本（极致内存优化版）
        
        通过多次前向传播，每次只获取部分层的注意力
        
        Args:
            messages: 对话消息列表
            max_length: 最大序列长度
            layers_per_batch: 每批处理的层数
            position_sample_rate: 位置采样率，减少需要分析的位置数
            
        Returns:
            样本级别的统计结果
        """
        full_text = self.parse_messages(messages)
        
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
        
        token_spans = self.identify_token_spans(full_text, input_ids)
        
        if not token_spans:
            return None
        
        sample_stats = {
            'think': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []},
            'action': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []},
            'post_response': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []}
        }
        
        # 找到需要分析的位置
        positions_by_phase = {'think': [], 'action': [], 'post_response': []}
        
        for span in token_spans:
            if span.phase in ['think', 'action']:
                phase_positions = list(range(span.start + 1, min(span.end, seq_len)))
                # 采样位置以减少计算量
                if position_sample_rate < 1.0 and len(phase_positions) > 10:
                    num_samples = max(5, int(len(phase_positions) * position_sample_rate))
                    phase_positions = list(np.linspace(
                        phase_positions[0], 
                        phase_positions[-1], 
                        num_samples, 
                        dtype=int
                    ))
                positions_by_phase[span.phase].extend(phase_positions)
        
        # Post-response positions
        obs_end_positions = [span.end for span in token_spans if span.phase == 'observation']
        for obs_end in obs_end_positions:
            if obs_end < seq_len:
                positions_by_phase['post_response'].append(obs_end)
        
        all_positions = []
        for phase, positions in positions_by_phase.items():
            for pos in positions:
                all_positions.append((pos, phase))
        
        if not all_positions:
            return None
        
        try:
            with torch.no_grad():
                # 一次性获取所有层的注意力
                outputs = self.model(
                    **inputs, 
                    output_attentions=True,
                    use_cache=False
                )
                attentions = outputs.attentions  # Tuple of tensors
                
                # 处理每个位置
                for pos, phase_key in all_positions:
                    # 遍历每一层
                    for layer_idx in range(self.num_layers):
                        # [batch, num_heads, seq_len, seq_len] -> [num_heads, pos+1]
                        layer_attn = attentions[layer_idx][0]
                        pos_attn = layer_attn[:, pos, :pos + 1].cpu()
                        entropy = self.compute_entropy(pos_attn)
                        sample_stats[phase_key]['entropy_by_layer'][layer_idx].append(entropy)
                    
                    # 注意力分配（使用最后一层）
                    last_layer_attn = attentions[-1][0][:, pos, :pos + 1].cpu()
                    think_score, obs_score, _ = self.compute_attention_allocation(
                        last_layer_attn, token_spans, pos
                    )
                    sample_stats[phase_key]['attn_think'].append(think_score)
                    sample_stats[phase_key]['attn_obs'].append(obs_score)
                
                # 清理
                del outputs, attentions
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM for sequence length {seq_len}, trying with reduced positions...")
                self.clear_memory()
                
                # 重试：更激进的采样
                return self.analyze_sample_minimal(messages, max_length)
            raise
        
        return sample_stats
    
    def analyze_sample_minimal(
        self,
        messages: List[Dict],
        max_length: int = 2048
    ) -> Optional[Dict]:
        """
        最小化内存使用的分析方法
        只分析少量关键位置和层
        """
        full_text = self.parse_messages(messages)
        
        # 截断到更短的长度
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        
        seq_len = inputs['input_ids'].shape[1]
        token_spans = self.identify_token_spans(full_text, inputs['input_ids'])
        
        if not token_spans:
            return None
        
        sample_stats = {
            'think': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []},
            'action': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []},
            'post_response': {'entropy_by_layer': defaultdict(list), 'attn_think': [], 'attn_obs': []}
        }
        
        # 只采样关键层：第一层、中间层、最后几层
        key_layers = [0, self.num_layers // 4, self.num_layers // 2, 
                      3 * self.num_layers // 4, self.num_layers - 1]
        key_layers = sorted(set(key_layers))
        
        # 每个phase只取3个位置
        positions_by_phase = {'think': [], 'action': [], 'post_response': []}
        
        for span in token_spans:
            if span.phase in ['think', 'action']:
                span_len = min(span.end, seq_len) - span.start
                if span_len > 0:
                    # 取开头、中间、结尾
                    positions = [
                        span.start + 1,
                        span.start + span_len // 2,
                        min(span.end - 1, seq_len - 1)
                    ]
                    positions = [p for p in positions if span.start < p < seq_len]
                    positions_by_phase[span.phase].extend(positions[:3])
        
        obs_end_positions = [span.end for span in token_spans if span.phase == 'observation']
        for obs_end in obs_end_positions[:2]:  # 最多2个
            if obs_end < seq_len:
                positions_by_phase['post_response'].append(obs_end)
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs, 
                    output_attentions=True,
                    use_cache=False
                )
                attentions = outputs.attentions
                
                for phase, positions in positions_by_phase.items():
                    for pos in positions:
                        for layer_idx in key_layers:
                            layer_attn = attentions[layer_idx][0]
                            pos_attn = layer_attn[:, pos, :pos + 1].cpu()
                            entropy = self.compute_entropy(pos_attn)
                            sample_stats[phase]['entropy_by_layer'][layer_idx].append(entropy)
                        
                        last_layer_attn = attentions[-1][0][:, pos, :pos + 1].cpu()
                        think_score, obs_score, _ = self.compute_attention_allocation(
                            last_layer_attn, token_spans, pos
                        )
                        sample_stats[phase]['attn_think'].append(think_score)
                        sample_stats[phase]['attn_obs'].append(obs_score)
                
                del outputs, attentions
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Still OOM, skipping sample")
                self.clear_memory()
                return None
            raise
        
        return sample_stats
    
    def analyze_sample(
        self,
        messages: List[Dict],
        max_length: int = 4096
    ) -> Optional[Dict]:
        """分析单个样本（自动选择策略）"""
        full_text = self.parse_messages(messages)
        estimated_tokens = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        
        # 根据序列长度选择策略
        if estimated_tokens > 3000:
            return self.analyze_sample_minimal(messages, max_length=2048)
        elif estimated_tokens > 1500:
            return self.analyze_sample_layerwise(
                messages, 
                max_length=max_length,
                position_sample_rate=0.2
            )
        else:
            return self.analyze_sample_layerwise(
                messages, 
                max_length=max_length,
                position_sample_rate=0.5
            )
    
    def analyze_dataset(
        self,
        json_path: str,
        max_samples: Optional[int] = None,
        save_interval: int = 50,
        checkpoint_dir: Optional[str] = None
    ):
        """
        分析整个数据集
        
        Args:
            json_path: JSON 文件路径
            max_samples: 最大样本数
            save_interval: 保存检查点的间隔
            checkpoint_dir: 检查点保存目录
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
                    for phase in ['think', 'action', 'post_response']:
                        for layer_idx, entropies in sample_stats[phase]['entropy_by_layer'].items():
                            self.stats[phase].entropy_by_layer[layer_idx].extend(entropies)
                        self.stats[phase].attention_to_think.extend(sample_stats[phase]['attn_think'])
                        self.stats[phase].attention_to_observation.extend(sample_stats[phase]['attn_obs'])
                    success_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Error processing sample {idx}: {e}")
                continue
            
            # 定期清理内存和保存检查点
            if (idx + 1) % 5 == 0:
                self.clear_memory()
            
            if checkpoint_dir and (idx + 1) % save_interval == 0:
                self._save_checkpoint(checkpoint_dir, idx)
                print(f"Checkpoint saved at index {idx}")
        
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
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        summary = {}
        
        for phase in ['think', 'action', 'post_response']:
            phase_summary = {
                'avg_entropy_by_layer': {},
                'std_entropy_by_layer': {},
                'avg_attention_to_think': 0.0,
                'avg_attention_to_observation': 0.0,
                'num_samples': 0
            }
            
            for layer_idx, entropies in self.stats[phase].entropy_by_layer.items():
                if entropies:
                    phase_summary['avg_entropy_by_layer'][layer_idx] = float(np.mean(entropies))
                    phase_summary['std_entropy_by_layer'][layer_idx] = float(np.std(entropies))
            
            if self.stats[phase].attention_to_think:
                phase_summary['avg_attention_to_think'] = float(np.mean(self.stats[phase].attention_to_think))
                phase_summary['avg_attention_to_observation'] = float(np.mean(self.stats[phase].attention_to_observation))
                phase_summary['num_samples'] = len(self.stats[phase].attention_to_think)
            
            summary[phase] = phase_summary
        
        return summary
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {'think': 'blue', 'action': 'red', 'post_response': 'green'}
        labels = {'think': 'Think Phase', 'action': 'Action Phase', 'post_response': 'Post-Response'}
        
        # 1. 各阶段各层的平均熵
        ax1 = axes[0, 0]
        for phase in ['think', 'action', 'post_response']:
            layers = sorted(self.stats[phase].entropy_by_layer.keys())
            avg_entropies = [np.mean(self.stats[phase].entropy_by_layer[l]) 
                           for l in layers if self.stats[phase].entropy_by_layer[l]]
            if avg_entropies and layers:
                ax1.plot(layers[:len(avg_entropies)], avg_entropies, 
                        color=colors[phase], label=labels[phase], linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Average Entropy', fontsize=12)
        ax1.set_title('Average Attention Entropy by Layer', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 深层网络的熵对比
        ax2 = axes[0, 1]
        bar_data = []
        for phase in ['think', 'action', 'post_response']:
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
        ax2.bar(x_pos, bar_data, color=[colors[p] for p in ['think', 'action', 'post_response']])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Think', 'Action', 'Post-Response'])
        ax2.set_ylabel('Average Entropy', fontsize=12)
        ax2.set_title('Deep Layers Average Entropy', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 注意力分配对比
        ax3 = axes[1, 0]
        phases = ['think', 'action', 'post_response']
        
        attn_think = [np.mean(self.stats[p].attention_to_think) if self.stats[p].attention_to_think else 0 
                     for p in phases]
        attn_obs = [np.mean(self.stats[p].attention_to_observation) if self.stats[p].attention_to_observation else 0 
                   for p in phases]
        
        x = np.arange(len(phases))
        width = 0.35
        
        ax3.bar(x - width/2, attn_think, width, label='Attention to Think', color='steelblue')
        ax3.bar(x + width/2, attn_obs, width, label='Attention to Observation', color='coral')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Think', 'Action', 'Post-Response'])
        ax3.set_ylabel('Attention Proportion', fontsize=12)
        ax3.set_title('Attention Allocation: Think vs Observation', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 样本数量统计
        ax4 = axes[1, 1]
        sample_counts = [len(self.stats[p].attention_to_think) for p in phases]
        ax4.bar(x, sample_counts, color=['blue', 'red', 'green'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Think', 'Action', 'Post-Response'])
        ax4.set_ylabel('Number of Positions Analyzed', fontsize=12)
        ax4.set_title('Analysis Coverage', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def save_results(self, save_path: str):
        """保存分析结果到 JSON"""
        summary = self.get_summary()
        
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
    max_samples: Optional[int] = None,
    device_map: str = "auto",
    max_memory: Optional[Dict] = None,
    load_in_4bit: bool = False
):
    """对比 Baseline 和 Ablation 模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 分析 Baseline
    print("=" * 60)
    print("Analyzing Baseline Model...")
    print("=" * 60)
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
    print("\n" + "=" * 60)
    print("Analyzing Ablation Model...")
    print("=" * 60)
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
    
    # 对比图
    plot_comparison(results, os.path.join(output_dir, "comparison_plots.png"))
    
    # 打印对比
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    for phase in ['think', 'action', 'post_response']:
        print(f"\n[{phase.upper()} PHASE]")
        
        b_attn_think = results['baseline'][phase]['avg_attention_to_think']
        a_attn_think = results['ablation'][phase]['avg_attention_to_think']
        b_attn_obs = results['baseline'][phase]['avg_attention_to_observation']
        a_attn_obs = results['ablation'][phase]['avg_attention_to_observation']
        
        print(f"  Attention to Think    - Baseline: {b_attn_think:.4f}, Ablation: {a_attn_think:.4f}, Δ: {a_attn_think - b_attn_think:+.4f}")
        print(f"  Attention to Observation - Baseline: {b_attn_obs:.4f}, Ablation: {a_attn_obs:.4f}, Δ: {a_attn_obs - b_attn_obs:+.4f}")
    
    return results


def plot_comparison(results: Dict, save_path: str):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'baseline': 'blue', 'ablation': 'orange'}
    phases = ['think', 'action', 'post_response']
    
    # 1. 熵对比 (Action Phase)
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
    ax2.set_xticklabels(['Think', 'Action', 'Post-Response'])
    ax2.set_ylabel('Deep Layer Entropy', fontsize=12)
    ax2.set_title('Deep Layer Entropy Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Attention to Think
    ax3 = axes[1, 0]
    baseline_attn = [results['baseline'][p]['avg_attention_to_think'] for p in phases]
    ablation_attn = [results['ablation'][p]['avg_attention_to_think'] for p in phases]
    
    ax3.bar(x - width/2, baseline_attn, width, label='Baseline', color=colors['baseline'])
    ax3.bar(x + width/2, ablation_attn, width, label='Ablation', color=colors['ablation'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Think', 'Action', 'Post-Response'])
    ax3.set_ylabel('Attention Score', fontsize=12)
    ax3.set_title('Attention to Think Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Attention to Observation
    ax4 = axes[1, 1]
    baseline_obs = [results['baseline'][p]['avg_attention_to_observation'] for p in phases]
    ablation_obs = [results['ablation'][p]['avg_attention_to_observation'] for p in phases]
    
    ax4.bar(x - width/2, baseline_obs, width, label='Baseline', color=colors['baseline'])
    ax4.bar(x + width/2, ablation_obs, width, label='Ablation', color=colors['ablation'])
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Think', 'Action', 'Post-Response'])
    ax4.set_ylabel('Attention Score', fontsize=12)
    ax4.set_title('Attention to Observation Comparison', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Entropy Analysis (Multi-GPU, Eager Attention)")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_path", type=str, required=True, help="JSON data path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare two models")
    parser.add_argument("--ablation_path", type=str, default=None, help="Ablation model path")
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
        
        summary = analyzer.get_summary()
        print("\n" + "=" * 60)
        print("Analysis Summary")
        print("=" * 60)
        
        for phase, stats in summary.items():
            print(f"\n[{phase.upper()}]")
            print(f"  Positions Analyzed: {stats['num_samples']}")
            print(f"  Avg Attention to Think: {stats['avg_attention_to_think']:.4f}")
            print(f"  Avg Attention to Observation: {stats['avg_attention_to_observation']:.4f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        analyzer.save_results(os.path.join(args.output_dir, "results.json"))
        analyzer.plot_results(os.path.join(args.output_dir, "plots.png"))