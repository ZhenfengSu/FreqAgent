"""
Attention Entropy Analysis - Memory Optimized Version
针对大模型 (32B+) 和长上下文的优化版本
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


class MemoryEfficientAttentionAnalyzer:
    """
    内存优化版注意力分析器
    
    核心优化策略：
    1. 使用 forward hooks 逐层提取注意力，避免同时存储所有层
    2. 只分析采样的层（如最后5层）
    3. 只分析采样的位置
    4. 立即计算统计量并释放原始注意力矩阵
    """
    
    def __init__(
        self,
        model_path: str,
        device_map: Union[str, Dict] = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_memory: Optional[Dict] = None,
        load_in_4bit: bool = False,  # 默认使用4bit量化
        layers_to_analyze: Optional[List[int]] = None,  # 只分析指定层
        analyze_last_n_layers: int = 8,  # 或只分析最后n层
    ):
        """
        初始化分析器
        """
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {mem:.1f} GB")
        
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
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # 必须使用eager以支持注意力输出
        }
        
        if load_in_4bit:
            print("Loading model in 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        
        # 设置内存限制
        if max_memory is None and torch.cuda.device_count() > 0:
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory
                # 保守分配，留出更多空间给注意力矩阵
                max_memory[i] = f"{int(total_mem * 0.70 / 1024**3)}GB"
            max_memory["cpu"] = "50GB"
            print(f"Memory allocation: {max_memory}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            max_memory=max_memory,
            **model_kwargs
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        
        # 确定要分析的层
        if layers_to_analyze is not None:
            self.layers_to_analyze = layers_to_analyze
        else:
            # 默认分析最后n层 + 一些早期层
            last_layers = list(range(self.num_layers - analyze_last_n_layers, self.num_layers))
            early_layers = [0, self.num_layers // 4, self.num_layers // 2]
            self.layers_to_analyze = sorted(set(early_layers + last_layers))
        
        print(f"Model loaded. Total layers: {self.num_layers}")
        print(f"Layers to analyze: {self.layers_to_analyze}")
        
        # 统计结果存储
        self.stats = {
            'think': AttentionStats(),
            'action': AttentionStats(),
            'post_response': AttentionStats()
        }
        
        # Hook 相关
        self._attention_cache = {}
        self._hooks = []
    
    def clear_memory(self):
        """清理GPU内存"""
        self._attention_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_attention_hook(self, layer_idx: int):
        """创建注意力提取 hook"""
        def hook(module, input, output):
            # output 通常是 (hidden_states, attention_weights, ...)
            # 注意力权重的位置可能因模型而异
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    # 只保存需要的位置的注意力，立即移到CPU
                    self._attention_cache[layer_idx] = attn_weights.detach().cpu()
        return hook
    
    def _register_hooks(self):
        """注册注意力提取 hooks"""
        self._remove_hooks()
        
        # 查找注意力层 - 适配不同模型架构
        for layer_idx in self.layers_to_analyze:
            layer = None
            
            # 尝试不同的模型架构
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'layers'):
                    # LLaMA, Qwen, Mistral 等
                    layer = self.model.model.layers[layer_idx].self_attn
                elif hasattr(self.model.model, 'decoder'):
                    # 某些 decoder 模型
                    layer = self.model.model.decoder.layers[layer_idx].self_attn
            elif hasattr(self.model, 'transformer'):
                # GPT-2 风格
                layer = self.model.transformer.h[layer_idx].attn
            
            if layer is not None:
                hook = layer.register_forward_hook(self._get_attention_hook(layer_idx))
                self._hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有 hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attention_cache.clear()
    
    def identify_token_spans(self, text: str, input_ids: torch.Tensor) -> List[TokenSpan]:
        """识别文本中不同阶段的 token 区间"""
        spans = []
        
        patterns = {
            'think': r'<think>(.*?)</think>',
            'action': r'<tool_call>(.*?)</tool_call>',
            'observation': r'<tool_response>(.*?)</tool_response>',
        }
        
        text_spans = []
        for phase, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                text_spans.append({
                    'start': match.start(),
                    'end': match.end(),
                    'phase': phase,
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
                if char_end == 0:
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
        """计算注意力分布的熵"""
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.float().clamp(min=1e-10)
        attention_weights = attention_weights / attention_weights.sum()
        entropy = -torch.sum(attention_weights * torch.log(attention_weights))
        
        return entropy.item()
    
    def compute_attention_allocation(
        self,
        attention_weights: torch.Tensor,
        token_spans: List[TokenSpan],
        current_pos: int
    ) -> Dict[str, float]:
        """计算注意力在不同区域的分配比例"""
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights[:current_pos + 1].float()
        
        region_scores = {'think': 0.0, 'observation': 0.0, 'action': 0.0, 'other': 0.0}
        region_lengths = {'think': 0, 'observation': 0, 'action': 0, 'other': 0}
        
        position_to_phase = {}
        for span in token_spans:
            if span.end <= current_pos + 1:
                for idx in range(span.start, min(span.end, current_pos + 1)):
                    position_to_phase[idx] = span.phase
        
        for idx in range(len(attention_weights)):
            weight = attention_weights[idx].item()
            phase = position_to_phase.get(idx, 'other')
            region_scores[phase] += weight
            region_lengths[phase] += 1
        
        normalized_scores = {}
        for phase in region_scores:
            if region_lengths[phase] > 0:
                normalized_scores[phase] = region_scores[phase] / region_lengths[phase]
            else:
                normalized_scores[phase] = 0.0
        
        return normalized_scores
    
    def parse_messages_for_context(self, messages: List[Dict]) -> str:
        """将完整 messages 转换为模型输入文本"""
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
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                text_parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                text_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                text_parts.append(f"<|assistant|>\n{content}")
        
        return "\n".join(text_parts)
    
    def analyze_sample_memory_efficient(
        self,
        messages: List[Dict],
        max_length: int = 32768,
        positions_per_phase: int = 10,  # 每个阶段采样的位置数
    ) -> Optional[Dict]:
        """
        内存优化版样本分析
        
        核心优化：
        1. 使用 hooks 逐层提取注意力
        2. 只保存需要分析的位置的注意力
        3. 立即计算统计量并释放
        """
        full_text = self.parse_messages_for_context(messages)
        
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
        
        # 找到需要分析的位置
        positions_by_phase = {'think': [], 'action': [], 'post_response': []}
        
        for span in token_spans:
            if span.phase == 'think':
                phase_positions = list(range(span.start + 1, min(span.end, seq_len)))
                positions_by_phase['think'].extend(phase_positions)
            elif span.phase == 'action':
                phase_positions = list(range(span.start + 1, min(span.end, seq_len)))
                positions_by_phase['action'].extend(phase_positions)
        
        for span in token_spans:
            if span.phase == 'observation':
                post_obs_start = span.end
                next_span_start = seq_len
                for next_span in token_spans:
                    if next_span.start > span.end:
                        next_span_start = next_span.start
                        break
                
                if post_obs_start < seq_len:
                    post_positions = list(range(post_obs_start, min(post_obs_start + 20, next_span_start, seq_len)))
                    positions_by_phase['post_response'].extend(post_positions)
        
        # 对每个阶段采样位置
        for phase in positions_by_phase:
            positions = positions_by_phase[phase]
            if len(positions) > positions_per_phase:
                indices = np.linspace(0, len(positions) - 1, positions_per_phase, dtype=int)
                positions_by_phase[phase] = [positions[i] for i in indices]
        
        total_positions = sum(len(p) for p in positions_by_phase.values())
        if total_positions == 0:
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
        
        try:
            # 注册 hooks
            self._register_hooks()
            
            with torch.no_grad():
                # 使用 output_attentions=True 进行前向传播
                # 但通过 hooks 只保存需要的层
                _ = self.model(
                    **inputs, 
                    output_attentions=True,
                    use_cache=False
                )
                
                # 处理每个阶段的位置
                for phase, positions in positions_by_phase.items():
                    for pos in positions:
                        if pos >= seq_len:
                            continue
                        
                        # 遍历已缓存的层
                        for layer_idx in self.layers_to_analyze:
                            if layer_idx not in self._attention_cache:
                                continue
                            
                            # [batch, num_heads, seq_len, seq_len]
                            layer_attn = self._attention_cache[layer_idx]
                            # [num_heads, pos+1]
                            pos_attn = layer_attn[0, :, pos, :pos + 1]
                            
                            entropy = self.compute_entropy(pos_attn)
                            sample_stats[phase]['entropy_by_layer'][layer_idx].append(entropy)
                        
                        # 计算注意力分配（使用最后一个分析的层）
                        last_analyzed_layer = max(self.layers_to_analyze)
                        if last_analyzed_layer in self._attention_cache:
                            last_layer_attn = self._attention_cache[last_analyzed_layer][0, :, pos, :pos + 1]
                            allocation = self.compute_attention_allocation(
                                last_layer_attn, token_spans, pos
                            )
                            sample_stats[phase]['attn_think'].append(allocation['think'])
                            sample_stats[phase]['attn_obs'].append(allocation['observation'])
                            sample_stats[phase]['attn_action'].append(allocation['action'])
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM for sequence length {seq_len}, skipping...")
                self.clear_memory()
                return None
            raise
        finally:
            self._remove_hooks()
            self.clear_memory()
        
        return sample_stats
    
    def analyze_sample_sliding_window(
        self,
        messages: List[Dict],
        max_length: int = 4096,
        window_size: int = 1024,  # 滑动窗口大小
        stride: int = 512,  # 步长
        positions_per_phase: int = 5,
    ) -> Optional[Dict]:
        """
        滑动窗口版本 - 用于超长序列
        
        将长序列分成多个窗口，分别分析后合并结果
        """
        full_text = self.parse_messages_for_context(messages)
        
        # 首先获取完整的 token ids 和 spans
        full_inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False
        )
        full_seq_len = full_inputs['input_ids'].shape[1]
        token_spans = self.identify_token_spans(full_text, full_inputs['input_ids'])
        
        if not token_spans:
            return None
        
        # 初始化统计
        sample_stats = {
            'think': {'entropy_by_layer': defaultdict(list), 
                     'attn_think': [], 'attn_obs': [], 'attn_action': []},
            'action': {'entropy_by_layer': defaultdict(list), 
                      'attn_think': [], 'attn_obs': [], 'attn_action': []},
            'post_response': {'entropy_by_layer': defaultdict(list), 
                             'attn_think': [], 'attn_obs': [], 'attn_action': []}
        }
        
        # 找到每个阶段的位置
        positions_by_phase = {'think': [], 'action': [], 'post_response': []}
        
        for span in token_spans:
            if span.phase == 'think':
                positions_by_phase['think'].extend(range(span.start + 1, min(span.end, full_seq_len)))
            elif span.phase == 'action':
                positions_by_phase['action'].extend(range(span.start + 1, min(span.end, full_seq_len)))
        
        for span in token_spans:
            if span.phase == 'observation':
                post_obs_start = span.end
                next_span_start = full_seq_len
                for next_span in token_spans:
                    if next_span.start > span.end:
                        next_span_start = next_span.start
                        break
                if post_obs_start < full_seq_len:
                    positions_by_phase['post_response'].extend(
                        range(post_obs_start, min(post_obs_start + 20, next_span_start, full_seq_len))
                    )
        
        # 采样位置
        for phase in positions_by_phase:
            positions = positions_by_phase[phase]
            if len(positions) > positions_per_phase:
                indices = np.linspace(0, len(positions) - 1, positions_per_phase, dtype=int)
                positions_by_phase[phase] = [positions[i] for i in indices]
        
        # 收集所有需要分析的位置
        all_target_positions = set()
        for positions in positions_by_phase.values():
            all_target_positions.update(positions)
        
        if not all_target_positions:
            return None
        
        first_device = next(self.model.parameters()).device
        
        # 使用滑动窗口处理
        for window_start in range(0, full_seq_len, stride):
            window_end = min(window_start + window_size, full_seq_len)
            
            # 检查这个窗口是否包含目标位置
            window_positions = [p for p in all_target_positions 
                              if window_start <= p < window_end]
            
            if not window_positions:
                continue
            
            # 提取窗口
            window_input_ids = full_inputs['input_ids'][:, window_start:window_end].to(first_device)
            window_attention_mask = full_inputs['attention_mask'][:, window_start:window_end].to(first_device)
            
            try:
                self._register_hooks()
                
                with torch.no_grad():
                    _ = self.model(
                        input_ids=window_input_ids,
                        attention_mask=window_attention_mask,
                        output_attentions=True,
                        use_cache=False
                    )
                
                # 处理窗口内的目标位置
                for phase, positions in positions_by_phase.items():
                    for global_pos in positions:
                        if global_pos < window_start or global_pos >= window_end:
                            continue
                        
                        local_pos = global_pos - window_start
                        
                        # 调整 token_spans 到窗口内的局部坐标
                        local_spans = []
                        for span in token_spans:
                            local_start = max(0, span.start - window_start)
                            local_end = min(window_end - window_start, span.end - window_start)
                            if local_end > local_start:
                                local_spans.append(TokenSpan(
                                    start=local_start,
                                    end=local_end,
                                    phase=span.phase
                                ))
                        
                        for layer_idx in self.layers_to_analyze:
                            if layer_idx not in self._attention_cache:
                                continue
                            
                            layer_attn = self._attention_cache[layer_idx]
                            pos_attn = layer_attn[0, :, local_pos, :local_pos + 1]
                            
                            entropy = self.compute_entropy(pos_attn)
                            sample_stats[phase]['entropy_by_layer'][layer_idx].append(entropy)
                        
                        last_analyzed_layer = max(self.layers_to_analyze)
                        if last_analyzed_layer in self._attention_cache:
                            last_layer_attn = self._attention_cache[last_analyzed_layer][0, :, local_pos, :local_pos + 1]
                            allocation = self.compute_attention_allocation(
                                last_layer_attn, local_spans, local_pos
                            )
                            sample_stats[phase]['attn_think'].append(allocation['think'])
                            sample_stats[phase]['attn_obs'].append(allocation['observation'])
                            sample_stats[phase]['attn_action'].append(allocation['action'])
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM in window [{window_start}:{window_end}], reducing window size...")
                    self.clear_memory()
                    continue
                raise
            finally:
                self._remove_hooks()
                self.clear_memory()
        
        return sample_stats
    
    def analyze_dataset(
        self,
        json_path: str,
        max_samples: Optional[int] = None,
        max_length: int = 4096,
        use_sliding_window: bool = False,
        window_size: int = 1024,
        save_interval: int = 20,
        checkpoint_dir: Optional[str] = None
    ):
        """分析整个数据集"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"Analyzing {len(data)} samples...")
        print(f"Max length: {max_length}, Use sliding window: {use_sliding_window}")
        
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
                if use_sliding_window:
                    sample_stats = self.analyze_sample_sliding_window(
                        messages, 
                        max_length=max_length,
                        window_size=window_size
                    )
                else:
                    sample_stats = self.analyze_sample_memory_efficient(
                        messages, 
                        max_length=max_length
                    )
                
                if sample_stats is not None:
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
            
            if (idx + 1) % 3 == 0:
                self.clear_memory()
            
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
            
            for layer_idx, entropies in self.stats[phase].entropy_by_layer.items():
                if entropies:
                    phase_summary['avg_entropy_by_layer'][layer_idx] = float(np.mean(entropies))
                    phase_summary['std_entropy_by_layer'][layer_idx] = float(np.std(entropies))
            
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
                            color=colors[phase], label=labels[phase], linewidth=2, marker='o', markersize=5)
        
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
                # 取最后几层
                last_layers = [l for l in layers if l >= max(layers) - 5]
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
        ax2.set_title('Deep Layers Entropy', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, bar_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Attention to Think
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
        
        # 4. Attention to Observation
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
        ax4.set_title('Attention to Observation', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 注意力分配对比
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
        """保存分析结果"""
        summary = self.get_summary()
        
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


def estimate_memory_requirement(num_layers: int, num_heads: int, seq_len: int, dtype_bytes: int = 2):
    """估算注意力矩阵的显存需求"""
    # 每层注意力矩阵: [batch, heads, seq, seq]
    per_layer_bytes = 1 * num_heads * seq_len * seq_len * dtype_bytes
    total_bytes = num_layers * per_layer_bytes
    
    print(f"Estimated attention memory for {num_layers} layers, {num_heads} heads, {seq_len} seq_len:")
    print(f"  Per layer: {per_layer_bytes / 1024**2:.1f} MB")
    print(f"  All layers: {total_bytes / 1024**3:.1f} GB")
    print(f"  Optimized (8 layers): {8 * per_layer_bytes / 1024**3:.1f} GB")
    
    return total_bytes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-Efficient Attention Entropy Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_path", type=str, required=True, help="JSON data path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--use_sliding_window", action="store_true", help="Use sliding window for long sequences")
    parser.add_argument("--window_size", type=int, default=1024, help="Sliding window size")
    parser.add_argument("--analyze_last_n_layers", type=int, default=8, help="Number of last layers to analyze")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--estimate_memory", action="store_true", help="Estimate memory requirement and exit")
    
    args = parser.parse_args()
    
    if args.estimate_memory:
        # 32B 模型典型配置
        estimate_memory_requirement(
            num_layers=64,  # 32B 通常有 64 层
            num_heads=40,   # 约 40 个头
            seq_len=args.max_length,
            dtype_bytes=2
        )
        exit(0)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer = MemoryEfficientAttentionAnalyzer(
        args.model_path,
        load_in_4bit=False,
        analyze_last_n_layers=args.analyze_last_n_layers,
    )
    
    analyzer.analyze_dataset(
        args.data_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        use_sliding_window=args.use_sliding_window,
        window_size=args.window_size,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoint")
    )
    
    # 保存和可视化结果
    summary = analyzer.get_summary()
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    for phase, stats in summary.items():
        print(f"\n[{phase.upper()}]")
        print(f"  Positions Analyzed: {stats['num_positions']}")
        print(f"  Avg Attention to Think: {stats['avg_attention_to_think']:.4f}")
        print(f"  Avg Attention to Observation: {stats['avg_attention_to_observation']:.4f}")
        print(f"  Avg Attention to Action: {stats['avg_attention_to_action']:.4f}")
        
        if stats['avg_entropy_by_layer']:
            layers = sorted([int(k) for k in stats['avg_entropy_by_layer'].keys()])
            last_layers = layers[-min(3, len(layers)):]
            last_entropy = np.mean([stats['avg_entropy_by_layer'][l] for l in last_layers])
            print(f"  Deep Layer Entropy: {last_entropy:.4f}")
    
    analyzer.save_results(os.path.join(args.output_dir, "results.json"))
    analyzer.plot_results(os.path.join(args.output_dir, "plots.png"))