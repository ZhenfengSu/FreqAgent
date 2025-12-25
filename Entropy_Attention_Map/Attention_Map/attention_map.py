"""
Attention Map Analyzer for Large Language Models
支持 32B 模型多卡运行，分析 <think>, <tool_call>, <tool_response> 等标签块的注意力分布
使用 register_hook 方式只提取指定层的 attention，节省显存
优化：支持长序列可视化、归一化 cross-region attention
"""

import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import gc

warnings.filterwarnings('ignore')


@dataclass
class TagSpan:
    """标签块的位置信息"""
    tag_type: str  # think, tool_call, tool_response, answer
    start_char: int
    end_char: int
    start_token: int = -1
    end_token: int = -1
    content: str = ""


@dataclass
class AttentionAnalysisResult:
    """注意力分析结果"""
    sample_id: int
    layer_id: int
    tag_spans: List[TagSpan]
    attention_matrix: np.ndarray
    token_texts: List[str]
    cross_region_attention: Dict[str, Dict[str, float]]
    # 新增：归一化后的 cross-region attention
    cross_region_attention_normalized: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SelfAttentionHook:
    """专门用于捕获 self-attention 输出的 Hook"""
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.attention = None
        self.handle = None
    
    def hook_fn(self, module, args, kwargs, output):
        """Hook 函数 - 使用 forward hook with kwargs"""
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                if attn_weights.dim() == 4:
                    self.attention = attn_weights.mean(dim=1).detach().float().cpu().numpy()
    
    def register(self, module):
        """注册 hook"""
        self.handle = module.register_forward_hook(
            self.hook_fn, 
            with_kwargs=True
        )
    
    def remove(self):
        """移除 hook"""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def clear(self):
        """清理"""
        self.attention = None


class AttentionCaptureContext:
    """上下文管理器，用于安全地注册和移除 hooks"""
    
    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.hooks: Dict[int, Any] = {}
        self.attention_modules = self._find_attention_modules()
    
    def _find_attention_modules(self) -> Dict[int, torch.nn.Module]:
        """找到模型中的 attention 模块"""
        attention_modules = {}
        
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        layers = None
        for attr in ['layers', 'decoder_layers', 'h', 'blocks']:
            if hasattr(base_model, attr):
                layers = getattr(base_model, attr)
                break
        
        if layers is None:
            raise ValueError("Cannot find transformer layers in model")
        
        for idx in self.layer_indices:
            if idx >= len(layers):
                print(f"Warning: Layer {idx} does not exist (model has {len(layers)} layers)")
                continue
            
            layer = layers[idx]
            attn_module = None
            for attr in ['self_attn', 'attention', 'attn', 'self_attention']:
                if hasattr(layer, attr):
                    attn_module = getattr(layer, attr)
                    break
            
            if attn_module is not None:
                attention_modules[idx] = attn_module
            else:
                print(f"Warning: Cannot find attention module in layer {idx}")
        
        return attention_modules
    
    def __enter__(self):
        """注册 hooks"""
        for layer_idx, module in self.attention_modules.items():
            hook = SelfAttentionHook(layer_idx)
            hook.register(module)
            self.hooks[layer_idx] = hook
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """移除 hooks 并清理"""
        for hook in self.hooks.values():
            hook.remove()
            hook.clear()
        self.hooks.clear()
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_attentions(self) -> Dict[int, np.ndarray]:
        """获取捕获的 attention maps"""
        result = {}
        for layer_idx, hook in self.hooks.items():
            if hook.attention is not None:
                result[layer_idx] = hook.attention[0]
        return result


class AttentionMapAnalyzer:
    """注意力图分析器 - 使用 Hook 方式提取 attention"""
    
    TAG_PATTERNS = {
        'think': r'<think>(.*?)</think>',
        'tool_call': r'<tool_call>(.*?)</tool_call>',
        'tool_response': r'<tool_response>(.*?)</tool_response>',
        'answer': r'<answer>(.*?)</answer>'
    }
    
    # 标签颜色配置
    TAG_COLORS = {
        'think': '#E74C3C',          # 红色
        'tool_call': '#27AE60',      # 绿色
        'tool_response': '#F39C12',  # 橙色
        'answer': '#9B59B6',         # 紫色
        'other': '#95A5A6'           # 灰色
    }
    
    DEFAULT_NUM_LAYERS = 4
    
    def __init__(
        self,
        model_path: str,
        selected_layers: List[int] = None,
        num_analysis_layers: int = 4,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 32768
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.torch_dtype = torch_dtype
        
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print(f"Loading model with device_map='{device_map}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.model.eval()
        
        self.num_layers = self._get_num_layers()
        print(f"Model has {self.num_layers} layers")
        
        if selected_layers is not None:
            self.selected_layers = selected_layers
        else:
            self.selected_layers = self._get_uniform_layers(num_analysis_layers)
        
        print(f"Selected layers for analysis: {self.selected_layers}")
    
    def _get_num_layers(self) -> int:
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        else:
            if hasattr(self.model, 'model'):
                base_model = self.model.model
            else:
                base_model = self.model
            
            for attr in ['layers', 'decoder_layers', 'h', 'blocks']:
                if hasattr(base_model, attr):
                    return len(getattr(base_model, attr))
            
            raise ValueError("Cannot determine number of layers")
    
    def _get_uniform_layers(self, num_layers: int = 4) -> List[int]:
        n = self.num_layers
        if num_layers >= n:
            return list(range(n))
        
        indices = []
        for i in range(num_layers):
            idx = int(i * (n - 1) / (num_layers - 1))
            indices.append(idx)
        
        return sorted(set(indices))
    
    def load_json_data(self, json_path: str) -> List[Dict]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {json_path}")
        return data
    
    def extract_assistant_content(self, messages: List[Dict]) -> str:
        assistant_contents = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if content:
                    assistant_contents.append(content)
        return '\n'.join(assistant_contents)
    
    def build_full_conversation(self, messages: List[Dict]) -> str:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"Warning: apply_chat_template failed: {e}")
        
        parts = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            parts.append(f"<|{role}|>\n{content}")
        return '\n'.join(parts)
    
    def find_tag_spans(self, text: str) -> List[TagSpan]:
        spans = []
        
        for tag_type, pattern in self.TAG_PATTERNS.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                span = TagSpan(
                    tag_type=tag_type,
                    start_char=match.start(),
                    end_char=match.end(),
                    content=match.group(1).strip()[:100]
                )
                spans.append(span)
        
        spans.sort(key=lambda x: x.start_char)
        return spans
    
    def map_char_to_token_positions(
        self, 
        text: str, 
        spans: List[TagSpan],
        encoding
    ) -> List[TagSpan]:
        token_offsets = []
        for i in range(len(encoding.input_ids[0])):
            try:
                char_span = encoding.token_to_chars(0, i)
                if char_span:
                    token_offsets.append((i, char_span.start, char_span.end))
            except:
                continue
        
        for span in spans:
            start_token = None
            end_token = None
            
            for token_idx, char_start, char_end in token_offsets:
                if start_token is None and char_end > span.start_char:
                    start_token = token_idx
                if char_start < span.end_char:
                    end_token = token_idx + 1
            
            if start_token is not None and end_token is not None:
                span.start_token = start_token
                span.end_token = end_token
        
        return spans
    
    def get_attention_maps_with_hooks(
        self, 
        input_ids: torch.Tensor
    ) -> Dict[int, np.ndarray]:
        with AttentionCaptureContext(self.model, self.selected_layers) as ctx:
            with torch.no_grad():
                _ = self.model(input_ids=input_ids)
            attention_maps = ctx.get_attentions()
        
        return attention_maps
    
    def analyze_cross_region_attention(
        self,
        attention_matrix: np.ndarray,
        tag_spans: List[TagSpan],
        seq_len: int
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        分析不同区域之间的注意力流向
        
        Returns:
            raw_attention: 原始平均注意力值
            normalized_attention: 归一化后的注意力值（多种归一化策略）
        """
        # 创建区域标记
        regions = {'other': []}
        token_to_region = ['other'] * seq_len
        
        for span in tag_spans:
            if span.start_token >= 0 and span.end_token >= 0:
                region_name = span.tag_type
                if region_name not in regions:
                    regions[region_name] = []
                
                for t in range(span.start_token, min(span.end_token, seq_len)):
                    token_to_region[t] = region_name
                    regions[region_name].append(t)
        
        regions['other'] = [i for i in range(seq_len) if token_to_region[i] == 'other']
        
        # 计算区域间注意力（原始值）
        raw_attention = {}
        region_names = list(regions.keys())
        
        # 收集所有 (src, tgt) 对的注意力值
        all_values = []
        pair_values = {}
        
        for src_region in region_names:
            raw_attention[src_region] = {}
            src_tokens = regions[src_region]
            
            if not src_tokens:
                continue
            
            for tgt_region in region_names:
                tgt_tokens = regions[tgt_region]
                if not tgt_tokens:
                    raw_attention[src_region][tgt_region] = 0.0
                    pair_values[(src_region, tgt_region)] = 0.0
                    continue
                
                # 计算从 src 到 tgt 的平均注意力
                total_attn = 0.0
                count = 0
                for s in src_tokens:
                    for t in tgt_tokens:
                        if t < s:  # causal attention
                            total_attn += attention_matrix[s, t]
                            count += 1
                
                avg_attn = total_attn / max(count, 1)
                raw_attention[src_region][tgt_region] = avg_attn
                pair_values[(src_region, tgt_region)] = avg_attn
                all_values.append(avg_attn)
        
        # 归一化策略
        normalized_attention = {}
        
        if all_values:
            all_values = np.array(all_values)
            non_zero_values = all_values[all_values > 0]
            
            if len(non_zero_values) > 0:
                # 策略1: Min-Max 归一化
                min_val = non_zero_values.min()
                max_val = non_zero_values.max()
                
                # 策略2: 对数归一化（处理数值范围大的情况）
                log_values = np.log1p(all_values * 1e6)  # 放大后取对数
                log_min = log_values.min()
                log_max = log_values.max()
                
                # 策略3: 按行归一化（每个 source region 的注意力分布）
                for src_region in region_names:
                    normalized_attention[src_region] = {}
                    
                    # 获取当前 source 的所有目标值
                    src_values = []
                    for tgt_region in region_names:
                        if src_region in raw_attention and tgt_region in raw_attention.get(src_region, {}):
                            src_values.append(raw_attention[src_region][tgt_region])
                    
                    src_values = np.array(src_values)
                    src_sum = src_values.sum()
                    
                    for tgt_region in region_names:
                        raw_val = raw_attention.get(src_region, {}).get(tgt_region, 0.0)
                        
                        # 使用行归一化（softmax-like）
                        if src_sum > 0:
                            norm_val = raw_val / src_sum
                        else:
                            norm_val = 0.0
                        
                        normalized_attention[src_region][tgt_region] = norm_val
        else:
            normalized_attention = raw_attention.copy()
        
        return raw_attention, normalized_attention
    
    def analyze_sample(
        self, 
        sample: Dict, 
        sample_id: int = 0
    ) -> List[AttentionAnalysisResult]:
        messages = sample.get('messages', [])
        full_text = self.build_full_conversation(messages)
        tag_spans = self.find_tag_spans(full_text)
        
        if not tag_spans:
            print(f"  Sample {sample_id}: No tags found")
            return []
        
        print(f"  Sample {sample_id}: Found {len(tag_spans)} tag spans")
        for span in tag_spans:
            print(f"    - {span.tag_type}: chars [{span.start_char}:{span.end_char}]")
        
        encoding = self.tokenizer(
            full_text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True
        )
        
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            first_device = next(iter(self.model.hf_device_map.values()))
            device = torch.device(first_device) if isinstance(first_device, str) else first_device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_ids = encoding.input_ids.to(device)
        seq_len = input_ids.shape[1]
        print(f"  Sequence length: {seq_len}")
        
        tag_spans = self.map_char_to_token_positions(full_text, tag_spans, encoding)
        valid_spans = [s for s in tag_spans if s.start_token >= 0 and s.end_token >= 0]
        print(f"  Valid spans with token positions: {len(valid_spans)}")
        for span in valid_spans:
            print(f"    - {span.tag_type}: tokens [{span.start_token}:{span.end_token}]")
        
        print(f"  Computing attention maps for layers {self.selected_layers} using hooks...")
        attention_maps = self.get_attention_maps_with_hooks(input_ids)
        
        if not attention_maps:
            print(f"  Warning: No attention maps captured. Trying alternative method...")
            attention_maps = self._fallback_get_attention(input_ids)
        
        print(f"  Captured attention from {len(attention_maps)} layers: {list(attention_maps.keys())}")
        
        token_ids = input_ids[0].cpu().tolist()
        token_texts = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        results = []
        for layer_idx, attn_matrix in attention_maps.items():
            raw_attn, norm_attn = self.analyze_cross_region_attention(
                attn_matrix, valid_spans, seq_len
            )
            
            result = AttentionAnalysisResult(
                sample_id=sample_id,
                layer_id=layer_idx,
                tag_spans=valid_spans,
                attention_matrix=attn_matrix,
                token_texts=token_texts,
                cross_region_attention=raw_attn,
                cross_region_attention_normalized=norm_attn
            )
            results.append(result)
        
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return results
    
    def _fallback_get_attention(self, input_ids: torch.Tensor) -> Dict[int, np.ndarray]:
        print("  Using fallback method with output_attentions...")
        attention_maps = {}
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True
            )
        
        if outputs.attentions is not None:
            for layer_idx in self.selected_layers:
                if layer_idx < len(outputs.attentions):
                    attn = outputs.attentions[layer_idx]
                    attn_avg = attn.mean(dim=1).float().cpu().numpy()
                    attention_maps[layer_idx] = attn_avg[0]
                    del attn
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return attention_maps
    
    def analyze_dataset(
        self, 
        json_path: str,
        max_samples: int = None
    ) -> List[AttentionAnalysisResult]:
        data = self.load_json_data(json_path)
        
        if max_samples:
            data = data[:max_samples]
        
        all_results = []
        for i, sample in enumerate(data):
            print(f"\nAnalyzing sample {i+1}/{len(data)}...")
            try:
                results = self.analyze_sample(sample, sample_id=i)
                all_results.extend(results)
            except Exception as e:
                print(f"  Error analyzing sample {i}: {e}")
                import traceback
                traceback.print_exc()
            
            gc.collect()
            torch.cuda.empty_cache()
        
        return all_results
    
    def _create_region_mask(
        self,
        seq_len: int,
        tag_spans: List[TagSpan]
    ) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]:
        """
        创建区域掩码和边界信息
        
        Returns:
            region_labels: 每个 token 的区域标签数组
            region_ranges: 每个区域的 (start, end) 范围列表
        """
        region_labels = np.array(['other'] * seq_len, dtype=object)
        region_ranges = defaultdict(list)
        
        for span in tag_spans:
            if span.start_token >= 0 and span.end_token >= 0:
                start = max(0, span.start_token)
                end = min(seq_len, span.end_token)
                region_labels[start:end] = span.tag_type
                region_ranges[span.tag_type].append((start, end))
        
        return region_labels, dict(region_ranges)
    
    def visualize_attention_heatmap(
        self,
        result: AttentionAnalysisResult,
        output_path: str = None,
        max_display_tokens: int = None,
        figsize: Tuple[int, int] = None,
        show_region_bars: bool = True,
        downsample_factor: int = None
    ):
        """
        可视化注意力热力图，标注标签区域
        
        针对长序列优化：
        1. 添加区域颜色条带
        2. 显示区域边界线
        3. 支持降采样显示
        4. 自动调整图像大小
        """
        attn = result.attention_matrix
        spans = result.tag_spans
        seq_len = attn.shape[0]
        
        # 自动决定是否降采样
        if downsample_factor is None:
            if seq_len > 8000:
                downsample_factor = max(1, seq_len // 2000)
            elif seq_len > 4000:
                downsample_factor = max(1, seq_len // 1000)
            else:
                downsample_factor = 1
        
        # 确定显示长度
        if max_display_tokens is None:
            max_display_tokens = seq_len
        
        display_len = min(seq_len, max_display_tokens)
        
        # 降采样处理
        if downsample_factor > 1:
            # 使用 block averaging 降采样
            new_len = display_len // downsample_factor
            attn_display = np.zeros((new_len, new_len))
            
            for i in range(new_len):
                for j in range(new_len):
                    i_start = i * downsample_factor
                    i_end = min((i + 1) * downsample_factor, display_len)
                    j_start = j * downsample_factor
                    j_end = min((j + 1) * downsample_factor, display_len)
                    attn_display[i, j] = attn[i_start:i_end, j_start:j_end].mean()
            
            display_len = new_len
            scale_factor = downsample_factor
        else:
            attn_display = attn[:display_len, :display_len]
            scale_factor = 1
        
        # 自动调整图像大小
        if figsize is None:
            base_size = min(20, max(12, display_len / 100))
            figsize = (base_size + 2, base_size)
        
        # 创建图像布局
        if show_region_bars:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                2, 2, 
                width_ratios=[1, 20], 
                height_ratios=[20, 1],
                wspace=0.02, 
                hspace=0.02
            )
            ax_main = fig.add_subplot(gs[0, 1])
            ax_left = fig.add_subplot(gs[0, 0])
            ax_bottom = fig.add_subplot(gs[1, 1])
            ax_corner = fig.add_subplot(gs[1, 0])
            ax_corner.axis('off')
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
        
        # 绘制主热力图
        im = ax_main.imshow(
            attn_display,
            cmap='Blues',
            aspect='auto',
            interpolation='nearest'
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # 创建区域标签
        region_labels, region_ranges = self._create_region_mask(seq_len, spans)
        
        # 绘制区域边界和高亮
        for tag_type, ranges in region_ranges.items():
            color = self.TAG_COLORS.get(tag_type, '#CCCCCC')
            
            for start, end in ranges:
                # 转换为显示坐标
                disp_start = start // scale_factor
                disp_end = end // scale_factor
                
                if disp_start < display_len and disp_end > 0:
                    disp_start = max(0, disp_start)
                    disp_end = min(display_len, disp_end)
                    
                    # 绘制边界框
                    rect = plt.Rectangle(
                        (disp_start - 0.5, disp_start - 0.5),
                        disp_end - disp_start,
                        disp_end - disp_start,
                        fill=False,
                        edgecolor=color,
                        linewidth=2.5,
                        linestyle='-',
                        zorder=10
                    )
                    ax_main.add_patch(rect)
                    
                    # 绘制水平和垂直参考线（区域边界）
                    ax_main.axhline(y=disp_start - 0.5, color=color, linewidth=1, alpha=0.5, linestyle='--')
                    ax_main.axhline(y=disp_end - 0.5, color=color, linewidth=1, alpha=0.5, linestyle='--')
                    ax_main.axvline(x=disp_start - 0.5, color=color, linewidth=1, alpha=0.5, linestyle='--')
                    ax_main.axvline(x=disp_end - 0.5, color=color, linewidth=1, alpha=0.5, linestyle='--')
        
        # 绘制区域颜色条
        if show_region_bars:
            # 创建颜色映射
            color_array = np.zeros((display_len, 1, 3))
            for i in range(display_len):
                orig_idx = min(i * scale_factor, seq_len - 1)
                tag = region_labels[orig_idx]
                color_hex = self.TAG_COLORS.get(tag, '#CCCCCC')
                # 转换 hex 到 RGB
                color_rgb = tuple(int(color_hex[i:i+2], 16) / 255 for i in (1, 3, 5))
                color_array[i, 0, :] = color_rgb
            
            # 左侧颜色条（Query）
            ax_left.imshow(color_array, aspect='auto', interpolation='nearest')
            ax_left.set_xticks([])
            ax_left.set_yticks([])
            ax_left.set_ylabel('Query (from)', fontsize=10)
            
            # 底部颜色条（Key）
            ax_bottom.imshow(
                color_array.transpose(1, 0, 2), 
                aspect='auto', 
                interpolation='nearest'
            )
            ax_bottom.set_xticks([])
            ax_bottom.set_yticks([])
            ax_bottom.set_xlabel('Key (attending to)', fontsize=10)
        
        # 设置标题
        title = f'Attention Map - Sample {result.sample_id}, Layer {result.layer_id}'
        title += f'\n(Sequence length: {seq_len}'
        if scale_factor > 1:
            title += f', downsampled {scale_factor}x'
        title += ')'
        ax_main.set_title(title, fontsize=12, fontweight='bold')
        
        # 添加图例
        legend_patches = [
            mpatches.Patch(color=color, label=f'<{tag}>')
            for tag, color in self.TAG_COLORS.items()
            if tag != 'other'
        ]
        ax_main.legend(
            handles=legend_patches, 
            loc='upper right',
            fontsize=8,
            framealpha=0.9
        )
        
        # 添加区域统计信息
        region_info = []
        for span in spans:
            if span.start_token >= 0:
                region_info.append(
                    f"<{span.tag_type}>: tokens {span.start_token}-{span.end_token}"
                )
        if region_info:
            info_text = '\n'.join(region_info)
            ax_main.text(
                0.02, 0.98, info_text,
                transform=ax_main.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention heatmap to {output_path}")
        
        plt.close()
        return fig
    
    def visualize_attention_heatmap_focused(
        self,
        result: AttentionAnalysisResult,
        output_path: str = None,
        figsize: Tuple[int, int] = (16, 14)
    ):
        """
        只显示标签区域的 attention 热力图（聚焦视图）
        适合长序列，只关注标签区域之间的注意力
        """
        attn = result.attention_matrix
        spans = result.tag_spans
        seq_len = attn.shape[0]
        
        if not spans:
            print("No tag spans to visualize")
            return None
        
        # 收集所有标签区域的 token
        region_tokens = {}
        for span in spans:
            if span.start_token >= 0 and span.end_token >= 0:
                key = f"{span.tag_type}_{span.start_token}"
                region_tokens[key] = {
                    'type': span.tag_type,
                    'start': span.start_token,
                    'end': span.end_token,
                    'tokens': list(range(span.start_token, min(span.end_token, seq_len)))
                }
        
        if not region_tokens:
            print("No valid tag spans")
            return None
        
        # 为每个区域计算代表性 attention（聚合）
        region_names = list(region_tokens.keys())
        n_regions = len(region_names)
        
        # 创建区域间 attention 矩阵
        region_attn = np.zeros((n_regions, n_regions))
        
        for i, (name_i, info_i) in enumerate(region_tokens.items()):
            for j, (name_j, info_j) in enumerate(region_tokens.items()):
                # 计算从区域 i 到区域 j 的平均 attention
                tokens_i = info_i['tokens']
                tokens_j = info_j['tokens']
                
                total = 0.0
                count = 0
                for ti in tokens_i:
                    for tj in tokens_j:
                        if tj < ti:  # causal
                            total += attn[ti, tj]
                            count += 1
                
                if count > 0:
                    region_attn[i, j] = total / count
        
        # 创建图
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：区域间 attention（原始值）
        ax1 = axes[0]
        labels = [f"{region_tokens[n]['type']}\n[{region_tokens[n]['start']}:{region_tokens[n]['end']}]" 
                  for n in region_names]
        
        # 使用对数刻度显示
        region_attn_display = region_attn.copy()
        region_attn_display[region_attn_display == 0] = np.nan
        
        sns.heatmap(
            region_attn_display,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='.2e',
            cmap='YlOrRd',
            ax=ax1,
            cbar_kws={'label': 'Attention (raw)'}
        )
        ax1.set_title('Region-to-Region Attention (Raw)', fontsize=11)
        ax1.set_xlabel('Key Region')
        ax1.set_ylabel('Query Region')
        
        # 右图：行归一化后的 attention
        ax2 = axes[1]
        row_sums = region_attn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        region_attn_norm = region_attn / row_sums
        
        sns.heatmap(
            region_attn_norm,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            ax=ax2,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Attention (row-normalized)'}
        )
        ax2.set_title('Region-to-Region Attention (Normalized)', fontsize=11)
        ax2.set_xlabel('Key Region')
        ax2.set_ylabel('Query Region')
        
        plt.suptitle(
            f'Focused Attention Analysis - Sample {result.sample_id}, Layer {result.layer_id}',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved focused attention heatmap to {output_path}")
        
        plt.close()
        return fig
    
    def visualize_cross_region_attention(
        self,
        results: List[AttentionAnalysisResult],
        output_path: str = None,
        figsize: Tuple[int, int] = None,
        use_normalized: bool = True,
        normalization_type: str = 'row'
    ):
        """
        可视化跨区域注意力流向（多层对比）
        
        Args:
            results: 分析结果列表
            output_path: 输出路径
            figsize: 图像大小
            use_normalized: 是否使用归一化值
            normalization_type: 归一化类型
                - 'row': 行归一化（每个 source 的注意力分布）
                - 'global': 全局归一化
                - 'log': 对数归一化
        """
        # 收集所有区域
        all_regions = set()
        for r in results:
            all_regions.update(r.cross_region_attention.keys())
        
        # 排序区域（保持一致的顺序）
        region_order = ['think', 'tool_call', 'tool_response', 'answer', 'other']
        regions = [r for r in region_order if r in all_regions]
        regions += [r for r in sorted(all_regions) if r not in region_order]
        
        # 按层组织数据
        layer_data = {}
        for r in results:
            layer_data[r.layer_id] = {
                'raw': r.cross_region_attention,
                'normalized': r.cross_region_attention_normalized
            }
        
        n_layers = len(layer_data)
        if n_layers == 0:
            print("No data to visualize")
            return None
        
        # 自动调整图像大小
        if figsize is None:
            figsize = (5 * n_layers + 2, 6)
        
        fig, axes = plt.subplots(1, n_layers, figsize=figsize)
        
        if n_layers == 1:
            axes = [axes]
        
        # 收集所有值用于全局归一化
        all_values = []
        for layer_id, data in layer_data.items():
            cross_attn = data['raw']
            for src in regions:
                for tgt in regions:
                    if src in cross_attn and tgt in cross_attn.get(src, {}):
                        val = cross_attn[src][tgt]
                        if val > 0:
                            all_values.append(val)
        
        all_values = np.array(all_values) if all_values else np.array([1.0])
        global_min = all_values.min() if len(all_values) > 0 else 0
        global_max = all_values.max() if len(all_values) > 0 else 1
        
        for idx, (layer_id, data) in enumerate(sorted(layer_data.items())):
            ax = axes[idx]
            
            if use_normalized and normalization_type == 'row':
                cross_attn = data['normalized']
                # 构建矩阵
                matrix = np.zeros((len(regions), len(regions)))
                for i, src in enumerate(regions):
                    for j, tgt in enumerate(regions):
                        if src in cross_attn and tgt in cross_attn.get(src, {}):
                            matrix[i, j] = cross_attn[src][tgt]
                
                # 绘制热力图
                sns.heatmap(
                    matrix,
                    xticklabels=regions,
                    yticklabels=regions,
                    annot=True,
                    fmt='.1%',
                    cmap='YlOrRd',
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cbar=idx == n_layers - 1,
                    cbar_kws={'label': 'Attention (row-normalized)'} if idx == n_layers - 1 else {}
                )
                
            elif normalization_type == 'log':
                cross_attn = data['raw']
                matrix = np.zeros((len(regions), len(regions)))
                for i, src in enumerate(regions):
                    for j, tgt in enumerate(regions):
                        if src in cross_attn and tgt in cross_attn.get(src, {}):
                            val = cross_attn[src][tgt]
                            matrix[i, j] = np.log1p(val * 1e6)  # 对数变换
                
                # 归一化到 0-1
                if matrix.max() > matrix.min():
                    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
                
                sns.heatmap(
                    matrix,
                    xticklabels=regions,
                    yticklabels=regions,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cbar=idx == n_layers - 1,
                    cbar_kws={'label': 'Attention (log-normalized)'} if idx == n_layers - 1 else {}
                )
                
            elif normalization_type == 'global':
                cross_attn = data['raw']
                matrix = np.zeros((len(regions), len(regions)))
                for i, src in enumerate(regions):
                    for j, tgt in enumerate(regions):
                        if src in cross_attn and tgt in cross_attn.get(src, {}):
                            val = cross_attn[src][tgt]
                            # 全局归一化
                            if global_max > global_min:
                                matrix[i, j] = (val - global_min) / (global_max - global_min)
                            else:
                                matrix[i, j] = 0
                
                sns.heatmap(
                    matrix,
                    xticklabels=regions,
                    yticklabels=regions,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cbar=idx == n_layers - 1,
                    cbar_kws={'label': 'Attention (global-normalized)'} if idx == n_layers - 1 else {}
                )
            else:
                # 原始值
                cross_attn = data['raw']
                matrix = np.zeros((len(regions), len(regions)))
                for i, src in enumerate(regions):
                    for j, tgt in enumerate(regions):
                        if src in cross_attn and tgt in cross_attn.get(src, {}):
                            matrix[i, j] = cross_attn[src][tgt]
                
                sns.heatmap(
                    matrix,
                    xticklabels=regions,
                    yticklabels=regions,
                    annot=True,
                    fmt='.2e',
                    cmap='YlOrRd',
                    ax=ax,
                    cbar=idx == n_layers - 1,
                    cbar_kws={'label': 'Attention (raw)'} if idx == n_layers - 1 else {}
                )
            
            ax.set_title(f'Layer {layer_id}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Target Region')
            if idx == 0:
                ax.set_ylabel('Source Region')
            
            # 为标签添加颜色
            for i, label in enumerate(ax.get_yticklabels()):
                label.set_color(self.TAG_COLORS.get(regions[i], 'black'))
                label.set_fontweight('bold')
            for i, label in enumerate(ax.get_xticklabels()):
                label.set_color(self.TAG_COLORS.get(regions[i], 'black'))
                label.set_fontweight('bold')
        
        norm_desc = {
            'row': 'Row-Normalized',
            'log': 'Log-Normalized', 
            'global': 'Global-Normalized',
            'raw': 'Raw Values'
        }
        plt.suptitle(
            f'Cross-Region Attention Flow ({norm_desc.get(normalization_type if use_normalized else "raw", "Custom")})',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved cross-region attention plot to {output_path}")
        
        plt.close()
        return fig
    
    def visualize_attention_flow_diagram(
        self,
        result: AttentionAnalysisResult,
        output_path: str = None,
        figsize: Tuple[int, int] = (12, 8),
        top_k_flows: int = 10
    ):
        """
        可视化注意力流向图（桑基图风格）
        显示各区域之间最强的注意力流向
        """
        spans = result.tag_spans
        cross_attn = result.cross_region_attention
        
        # 收集所有区域
        regions = list(cross_attn.keys())
        
        # 收集所有流向及其强度
        flows = []
        for src in regions:
            for tgt in regions:
                if src in cross_attn and tgt in cross_attn.get(src, {}):
                    attn_val = cross_attn[src][tgt]
                    if attn_val > 0:
                        flows.append((src, tgt, attn_val))
        
        if not flows:
            print("No attention flows to visualize")
            return None
        
        # 按注意力强度排序，取 top_k
        flows.sort(key=lambda x: x[2], reverse=True)
        top_flows = flows[:top_k_flows]
        
        # 归一化流量强度用于可视化
        max_flow = max(f[2] for f in top_flows)
        min_flow = min(f[2] for f in top_flows)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置区域位置
        unique_regions = sorted(set([f[0] for f in top_flows] + [f[1] for f in top_flows]))
        n_regions = len(unique_regions)
        
        # 使用两列布局
        region_positions = {}
        for i, region in enumerate(unique_regions):
            y = 1 - (i + 0.5) / n_regions
            region_positions[region] = y
        
        # 绘制区域节点
        for region, y in region_positions.items():
            color = self.TAG_COLORS.get(region, '#CCCCCC')
            
            # 左侧节点（Source）
            ax.add_patch(plt.Rectangle(
                (0.1, y - 0.03), 0.15, 0.06,
                facecolor=color, edgecolor='black', linewidth=2
            ))
            ax.text(0.175, y, region, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # 右侧节点（Target）
            ax.add_patch(plt.Rectangle(
                (0.75, y - 0.03), 0.15, 0.06,
                facecolor=color, edgecolor='black', linewidth=2
            ))
            ax.text(0.825, y, region, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 绘制流向箭头
        for src, tgt, attn_val in top_flows:
            src_y = region_positions[src]
            tgt_y = region_positions[tgt]
            
            # 线宽与注意力强度成正比
            if max_flow > min_flow:
                normalized = (attn_val - min_flow) / (max_flow - min_flow)
            else:
                normalized = 1.0
            
            linewidth = 1 + normalized * 8
            alpha = 0.3 + normalized * 0.6
            
            # 绘制曲线箭头
            from matplotlib.patches import FancyArrowPatch
            from matplotlib.path import Path
            import matplotlib.patches as mpatches
            
            arrow = FancyArrowPatch(
                (0.25, src_y), (0.75, tgt_y),
                connectionstyle=f"arc3,rad={0.2 if src_y != tgt_y else 0}",
                arrowstyle='->,head_length=10,head_width=6',
                linewidth=linewidth,
                color=self.TAG_COLORS.get(src, '#666666'),
                alpha=alpha
            )
            ax.add_patch(arrow)
            
            # 添加注意力值标签
            mid_x = 0.5
            mid_y = (src_y + tgt_y) / 2
            ax.text(
                mid_x, mid_y, f'{attn_val:.2e}',
                ha='center', va='center',
                fontsize=7, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title(
            f'Attention Flow Diagram - Sample {result.sample_id}, Layer {result.layer_id}\n(Top {len(top_flows)} flows)',
            fontsize=12,
            fontweight='bold'
        )
        
        # 添加图例
        ax.text(0.175, 1.05, 'Source', ha='center', fontsize=11, fontweight='bold')
        ax.text(0.825, 1.05, 'Target', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention flow diagram to {output_path}")
        
        plt.close()
        return fig
    
    def generate_summary_report(
        self,
        results: List[AttentionAnalysisResult]
    ) -> str:
        """生成分析摘要报告"""
        report_lines = [
            "=" * 70,
            "ATTENTION MAP ANALYSIS REPORT",
            "=" * 70,
            f"Model: {self.model_path}",
            f"Analyzed layers: {self.selected_layers}",
            ""
        ]
        
        samples = defaultdict(list)
        for r in results:
            samples[r.sample_id].append(r)
        
        for sample_id, sample_results in sorted(samples.items()):
            report_lines.append(f"\n{'='*50}")
            report_lines.append(f"SAMPLE {sample_id}")
            report_lines.append(f"{'='*50}")
            
            if sample_results:
                spans = sample_results[0].tag_spans
                seq_len = sample_results[0].attention_matrix.shape[0]
                report_lines.append(f"\nSequence Length: {seq_len}")
                report_lines.append(f"Tag Spans Found: {len(spans)}")
                for span in spans:
                    report_lines.append(
                        f"  - <{span.tag_type}>: tokens [{span.start_token}:{span.end_token}] "
                        f"({span.end_token - span.start_token} tokens)"
                    )
            
            report_lines.append("\nCross-Region Attention Summary:")
            report_lines.append("-" * 40)
            
            for r in sample_results:
                report_lines.append(f"\n  Layer {r.layer_id}:")
                
                # 使用归一化后的值
                norm_attn = r.cross_region_attention_normalized
                
                # 找出每个区域最关注哪里
                for src_region in ['think', 'tool_call', 'tool_response', 'answer']:
                    if src_region in norm_attn:
                        targets = norm_attn[src_region]
                        if targets:
                            sorted_targets = sorted(
                                targets.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )
                            top_3 = sorted_targets[:3]
                            top_str = ', '.join([f"{t}: {v:.1%}" for t, v in top_3])
                            report_lines.append(f"    {src_region} -> [{top_str}]")
                
                # 原始值统计
                raw_attn = r.cross_region_attention
                max_attn = 0
                max_flow = ""
                for src, targets in raw_attn.items():
                    for tgt, attn in targets.items():
                        if attn > max_attn:
                            max_attn = attn
                            max_flow = f"{src} -> {tgt}"
                
                report_lines.append(f"    [Strongest raw flow: {max_flow} ({max_attn:.2e})]")
        
        report = "\n".join(report_lines)
        return report


def main():
    """主函数"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Analyze attention maps for LLM')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, default='./attention_analysis',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=5,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices to analyze (e.g., "0,16,31,63")')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers to analyze if --layers not specified')
    parser.add_argument('--max_length', type=int, default=32768,
                       help='Maximum sequence length')
    parser.add_argument('--normalization', type=str, default='row',
                       choices=['row', 'log', 'global', 'raw'],
                       help='Normalization type for cross-region attention')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    selected_layers = None
    if args.layers:
        selected_layers = [int(x.strip()) for x in args.layers.split(',')]
    
    # 初始化分析器
    analyzer = AttentionMapAnalyzer(
        model_path=args.model_path,
        selected_layers=selected_layers,
        num_analysis_layers=args.num_layers,
        max_length=args.max_length
    )
    
    # 分析数据集
    results = analyzer.analyze_dataset(
        json_path=args.json_path,
        max_samples=args.max_samples
    )
    
    if not results:
        print("No results generated!")
        return
    
    # 生成可视化
    print("\nGenerating visualizations...")
    
    # 为每个样本的每一层生成多种可视化
    for r in results:
        # 1. 完整热力图（带区域标注）
        output_path = os.path.join(
            args.output_dir, 
            f"attention_heatmap_sample{r.sample_id}_layer{r.layer_id}.png"
        )
        analyzer.visualize_attention_heatmap(r, output_path=output_path)
        
        # 2. 聚焦视图（只显示标签区域）
        output_path = os.path.join(
            args.output_dir,
            f"attention_focused_sample{r.sample_id}_layer{r.layer_id}.png"
        )
        analyzer.visualize_attention_heatmap_focused(r, output_path=output_path)
        
        # 3. 注意力流向图
        output_path = os.path.join(
            args.output_dir,
            f"attention_flow_sample{r.sample_id}_layer{r.layer_id}.png"
        )
        analyzer.visualize_attention_flow_diagram(r, output_path=output_path)
    
    # 按样本分组生成跨区域注意力对比图
    by_sample = defaultdict(list)
    for r in results:
        by_sample[r.sample_id].append(r)
    
    for sample_id, sample_results in by_sample.items():
        # 使用不同的归一化方式
        for norm_type in ['row', 'log', 'global']:
            output_path = os.path.join(
                args.output_dir,
                f"cross_region_attention_sample{sample_id}_{norm_type}.png"
            )
            analyzer.visualize_cross_region_attention(
                sample_results, 
                output_path=output_path,
                use_normalized=True,
                normalization_type=norm_type
            )
    
    # 生成报告
    report = analyzer.generate_summary_report(results)
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
