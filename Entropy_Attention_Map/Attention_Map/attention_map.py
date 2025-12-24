"""
Attention Map Analyzer for Large Language Models
支持 32B 模型多卡运行，分析 <think>, <tool_call>, <tool_response> 等标签块的注意力分布
使用 register_hook 方式只提取指定层的 attention，节省显存
"""

import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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


class AttentionHook:
    """用于捕获指定层 attention 的 Hook 类"""
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.attention = None
        self.handle = None
    
    def hook_fn(self, module, input, output):
        """Hook 函数，捕获 attention weights"""
        # 不同模型架构的输出格式可能不同
        # 通常 attention 模块的输出是 (hidden_states, attention_weights, ...)
        # 或者是一个包含 attention_weights 的 tuple/namedtuple
        
        if isinstance(output, tuple):
            # 尝试找到 attention weights
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    # Shape: (batch, num_heads, seq_len, seq_len)
                    # 检查是否是方阵（attention weights 特征）
                    if item.shape[-1] == item.shape[-2]:
                        # 平均所有头，转换为 CPU numpy，立即释放 GPU 内存
                        self.attention = item.mean(dim=1).detach().float().cpu().numpy()
                        return
        elif hasattr(output, 'attentions') and output.attentions is not None:
            attn = output.attentions
            self.attention = attn.mean(dim=1).detach().float().cpu().numpy()
    
    def register(self, module):
        """注册 hook"""
        self.handle = module.register_forward_hook(self.hook_fn)
    
    def remove(self):
        """移除 hook"""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def clear(self):
        """清理捕获的 attention"""
        self.attention = None


class SelfAttentionHook:
    """专门用于捕获 self-attention 输出的 Hook"""
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.attention = None
        self.handle = None
    
    def hook_fn(self, module, args, kwargs, output):
        """Hook 函数 - 使用 forward hook with kwargs"""
        # 处理不同的输出格式
        if isinstance(output, tuple) and len(output) >= 2:
            # 通常第二个元素是 attention weights
            attn_weights = output[1]
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                if attn_weights.dim() == 4:  # (batch, heads, seq, seq)
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
        
        # 尝试不同的模型架构
        # Qwen2, LLaMA, Mistral 等常见架构
        
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        # 查找 layers
        layers = None
        for attr in ['layers', 'decoder_layers', 'h', 'blocks']:
            if hasattr(base_model, attr):
                layers = getattr(base_model, attr)
                break
        
        if layers is None:
            raise ValueError("Cannot find transformer layers in model")
        
        # 查找每层的 attention 模块
        for idx in self.layer_indices:
            if idx >= len(layers):
                print(f"Warning: Layer {idx} does not exist (model has {len(layers)} layers)")
                continue
            
            layer = layers[idx]
            
            # 尝试找到 self-attention 模块
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
                result[layer_idx] = hook.attention[0]  # 取 batch 的第一个
        return result


class AttentionMapAnalyzer:
    """注意力图分析器 - 使用 Hook 方式提取 attention"""
    
    # 需要提取的标签模式
    TAG_PATTERNS = {
        'think': r'<think>(.*?)</think>',
        'tool_call': r'<tool_call>(.*?)</tool_call>',
        'tool_response': r'<tool_response>(.*?)</tool_response>',
        'answer': r'<answer>(.*?)</answer>'
    }
    
    # 默认分析 4 层
    DEFAULT_NUM_LAYERS = 4
    
    def __init__(
        self,
        model_path: str,
        selected_layers: List[int] = None,
        num_analysis_layers: int = 4,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096
    ):
        """
        初始化分析器
        
        Args:
            model_path: 模型路径
            selected_layers: 要分析的层索引列表（如果指定，则忽略 num_analysis_layers）
            num_analysis_layers: 要分析的层数（默认4层，均匀分布）
            device_map: 设备映射策略，支持多卡
            torch_dtype: 模型精度
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.max_length = max_length
        self.torch_dtype = torch_dtype
        
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print(f"Loading model with device_map='{device_map}'...")
        # 注意：不需要 output_attentions=True，我们使用 hook
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager"  # 使用 eager 模式以获取完整 attention
        )
        self.model.eval()
        
        # 获取模型层数
        self.num_layers = self._get_num_layers()
        print(f"Model has {self.num_layers} layers")
        
        # 设置要分析的层（默认选择 4 层，均匀分布）
        if selected_layers is not None:
            self.selected_layers = selected_layers
        else:
            self.selected_layers = self._get_uniform_layers(num_analysis_layers)
        
        print(f"Selected layers for analysis: {self.selected_layers}")
    
    def _get_num_layers(self) -> int:
        """获取模型层数"""
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        else:
            # 尝试从模型结构获取
            if hasattr(self.model, 'model'):
                base_model = self.model.model
            else:
                base_model = self.model
            
            for attr in ['layers', 'decoder_layers', 'h', 'blocks']:
                if hasattr(base_model, attr):
                    return len(getattr(base_model, attr))
            
            raise ValueError("Cannot determine number of layers")
    
    def _get_uniform_layers(self, num_layers: int = 4) -> List[int]:
        """获取均匀分布的层索引"""
        n = self.num_layers
        if num_layers >= n:
            return list(range(n))
        
        # 均匀分布：包含第一层和最后一层
        indices = []
        for i in range(num_layers):
            idx = int(i * (n - 1) / (num_layers - 1))
            indices.append(idx)
        
        return sorted(set(indices))
    
    def load_json_data(self, json_path: str) -> List[Dict]:
        """加载JSON格式的数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {json_path}")
        return data
    
    def extract_assistant_content(self, messages: List[Dict]) -> str:
        """提取所有assistant角色的内容并拼接"""
        assistant_contents = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if content:
                    assistant_contents.append(content)
        return '\n'.join(assistant_contents)
    
    def build_full_conversation(self, messages: List[Dict]) -> str:
        """构建完整的对话文本（用于tokenization）"""
        # 使用 chat template 如果可用
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"Warning: apply_chat_template failed: {e}")
        
        # 回退到简单拼接
        parts = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            parts.append(f"<|{role}|>\n{content}")
        return '\n'.join(parts)
    
    def find_tag_spans(self, text: str) -> List[TagSpan]:
        """在文本中找到所有标签块的位置"""
        spans = []
        
        for tag_type, pattern in self.TAG_PATTERNS.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                span = TagSpan(
                    tag_type=tag_type,
                    start_char=match.start(),
                    end_char=match.end(),
                    content=match.group(1).strip()[:100]  # 保存前100字符作为预览
                )
                spans.append(span)
        
        # 按起始位置排序
        spans.sort(key=lambda x: x.start_char)
        return spans
    
    def map_char_to_token_positions(
        self, 
        text: str, 
        spans: List[TagSpan],
        encoding
    ) -> List[TagSpan]:
        """将字符位置映射到token位置"""
        # 获取每个token的字符偏移
        token_offsets = []
        for i in range(len(encoding.input_ids[0])):
            try:
                char_span = encoding.token_to_chars(0, i)
                if char_span:
                    token_offsets.append((i, char_span.start, char_span.end))
            except:
                continue
        
        # 为每个span找到对应的token范围
        for span in spans:
            start_token = None
            end_token = None
            
            for token_idx, char_start, char_end in token_offsets:
                # 找起始token
                if start_token is None and char_end > span.start_char:
                    start_token = token_idx
                # 找结束token
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
        """使用 hooks 获取指定层的注意力图"""
        
        # 使用上下文管理器安全地注册和移除 hooks
        with AttentionCaptureContext(self.model, self.selected_layers) as ctx:
            with torch.no_grad():
                # 只需要前向传播，不需要 output_attentions
                _ = self.model(input_ids=input_ids)
            
            # 获取捕获的 attention maps
            attention_maps = ctx.get_attentions()
        
        return attention_maps
    
    def analyze_cross_region_attention(
        self,
        attention_matrix: np.ndarray,
        tag_spans: List[TagSpan],
        seq_len: int
    ) -> Dict[str, Dict[str, float]]:
        """分析不同区域之间的注意力流向"""
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
        
        # 填充other区域
        regions['other'] = [i for i in range(seq_len) if token_to_region[i] == 'other']
        
        # 计算区域间注意力
        cross_attention = {}
        region_names = list(regions.keys())
        
        for src_region in region_names:
            cross_attention[src_region] = {}
            src_tokens = regions[src_region]
            
            if not src_tokens:
                continue
            
            for tgt_region in region_names:
                tgt_tokens = regions[tgt_region]
                if not tgt_tokens:
                    cross_attention[src_region][tgt_region] = 0.0
                    continue
                
                # 计算从 src 到 tgt 的平均注意力
                total_attn = 0.0
                count = 0
                for s in src_tokens:
                    for t in tgt_tokens:
                        if t < s:  # causal attention
                            total_attn += attention_matrix[s, t]
                            count += 1
                
                cross_attention[src_region][tgt_region] = total_attn / max(count, 1)
        
        return cross_attention
    
    def analyze_sample(
        self, 
        sample: Dict, 
        sample_id: int = 0
    ) -> List[AttentionAnalysisResult]:
        """分析单个样本"""
        messages = sample.get('messages', [])
        
        # 构建完整对话
        full_text = self.build_full_conversation(messages)
        
        # 在完整文本中找标签位置
        tag_spans = self.find_tag_spans(full_text)
        
        if not tag_spans:
            print(f"  Sample {sample_id}: No tags found")
            return []
        
        print(f"  Sample {sample_id}: Found {len(tag_spans)} tag spans")
        for span in tag_spans:
            print(f"    - {span.tag_type}: chars [{span.start_char}:{span.end_char}]")
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True
        )
        
        # 获取第一个可用的 GPU 设备
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # 多卡情况，获取第一个设备
            first_device = next(iter(self.model.hf_device_map.values()))
            device = torch.device(first_device) if isinstance(first_device, str) else first_device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_ids = encoding.input_ids.to(device)
        seq_len = input_ids.shape[1]
        print(f"  Sequence length: {seq_len}")
        
        # 映射token位置
        tag_spans = self.map_char_to_token_positions(full_text, tag_spans, encoding)
        
        # 过滤有效的spans
        valid_spans = [s for s in tag_spans if s.start_token >= 0 and s.end_token >= 0]
        print(f"  Valid spans with token positions: {len(valid_spans)}")
        for span in valid_spans:
            print(f"    - {span.tag_type}: tokens [{span.start_token}:{span.end_token}]")
        
        # 使用 hooks 获取 attention maps
        print(f"  Computing attention maps for layers {self.selected_layers} using hooks...")
        attention_maps = self.get_attention_maps_with_hooks(input_ids)
        
        if not attention_maps:
            print(f"  Warning: No attention maps captured. Trying alternative method...")
            attention_maps = self._fallback_get_attention(input_ids)
        
        print(f"  Captured attention from {len(attention_maps)} layers: {list(attention_maps.keys())}")
        
        # 获取token文本（用于可视化）
        token_ids = input_ids[0].cpu().tolist()
        token_texts = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        # 分析每一层
        results = []
        for layer_idx, attn_matrix in attention_maps.items():
            cross_attn = self.analyze_cross_region_attention(
                attn_matrix, valid_spans, seq_len
            )
            
            result = AttentionAnalysisResult(
                sample_id=sample_id,
                layer_id=layer_idx,
                tag_spans=valid_spans,
                attention_matrix=attn_matrix,
                token_texts=token_texts,
                cross_region_attention=cross_attn
            )
            results.append(result)
        
        # 清理 GPU 内存
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return results
    
    def _fallback_get_attention(self, input_ids: torch.Tensor) -> Dict[int, np.ndarray]:
        """备用方法：使用 output_attentions 但只保留需要的层"""
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
                    # 立即删除原始 tensor
                    del attn
        
        # 清理
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return attention_maps
    
    def analyze_dataset(
        self, 
        json_path: str,
        max_samples: int = None
    ) -> List[AttentionAnalysisResult]:
        """分析整个数据集"""
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
            
            # 每个样本后清理内存
            gc.collect()
            torch.cuda.empty_cache()
        
        return all_results
    
    def visualize_attention_heatmap(
        self,
        result: AttentionAnalysisResult,
        output_path: str = None,
        max_display_tokens: int = 200,
        figsize: Tuple[int, int] = (16, 14)
    ):
        """可视化注意力热力图，标注标签区域"""
        attn = result.attention_matrix
        spans = result.tag_spans
        
        # 限制显示的token数量
        display_len = min(attn.shape[0], max_display_tokens)
        attn_display = attn[:display_len, :display_len]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(
            attn_display,
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        # 标注标签区域
        colors = {
            'think': 'red',
            'tool_call': 'green',
            'tool_response': 'orange',
            'answer': 'purple'
        }
        
        for span in spans:
            if span.start_token < display_len and span.end_token <= display_len:
                color = colors.get(span.tag_type, 'gray')
                
                # 画矩形框标注区域
                rect = plt.Rectangle(
                    (span.start_token, span.start_token),
                    span.end_token - span.start_token,
                    span.end_token - span.start_token,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                    linestyle='--'
                )
                ax.add_patch(rect)
                
                # 添加标签
                ax.annotate(
                    f'<{span.tag_type}>',
                    xy=(span.start_token, span.start_token),
                    fontsize=8,
                    color=color,
                    fontweight='bold'
                )
        
        ax.set_title(f'Attention Map - Sample {result.sample_id}, Layer {result.layer_id}')
        ax.set_xlabel('Key Position (attending to)')
        ax.set_ylabel('Query Position (from)')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color=c, linewidth=2, linestyle='--', label=f'<{t}>')
            for t, c in colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention heatmap to {output_path}")
        
        plt.close()
        return fig
    
    def visualize_cross_region_attention(
        self,
        results: List[AttentionAnalysisResult],
        output_path: str = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """可视化跨区域注意力流向（多层对比）"""
        # 收集所有区域
        all_regions = set()
        for r in results:
            all_regions.update(r.cross_region_attention.keys())
        regions = sorted(all_regions)
        
        # 按层组织数据
        layer_data = defaultdict(dict)
        for r in results:
            layer_data[r.layer_id] = r.cross_region_attention
        
        n_layers = len(layer_data)
        if n_layers == 0:
            print("No data to visualize")
            return None
        
        fig, axes = plt.subplots(1, n_layers, figsize=figsize)
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_id, cross_attn) in enumerate(sorted(layer_data.items())):
            # 构建矩阵
            matrix = np.zeros((len(regions), len(regions)))
            for i, src in enumerate(regions):
                for j, tgt in enumerate(regions):
                    if src in cross_attn and tgt in cross_attn[src]:
                        matrix[i, j] = cross_attn[src][tgt]
            
            ax = axes[idx]
            sns.heatmap(
                matrix,
                xticklabels=regions,
                yticklabels=regions,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                ax=ax,
                cbar=idx == n_layers - 1  # 只在最后一个显示colorbar
            )
            ax.set_title(f'Layer {layer_id}')
            ax.set_xlabel('Target Region')
            ax.set_ylabel('Source Region')
        
        plt.suptitle('Cross-Region Attention Flow', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved cross-region attention plot to {output_path}")
        
        plt.close()
        return fig
    
    def generate_summary_report(
        self,
        results: List[AttentionAnalysisResult]
    ) -> str:
        """生成分析摘要报告"""
        report_lines = [
            "=" * 60,
            "ATTENTION MAP ANALYSIS REPORT",
            "=" * 60,
            f"Model: {self.model_path}",
            f"Analyzed layers: {self.selected_layers}",
            ""
        ]
        
        # 按样本和层组织结果
        samples = defaultdict(list)
        for r in results:
            samples[r.sample_id].append(r)
        
        for sample_id, sample_results in sorted(samples.items()):
            report_lines.append(f"\n{'='*40}")
            report_lines.append(f"SAMPLE {sample_id}")
            report_lines.append(f"{'='*40}")
            
            # 显示标签信息
            if sample_results:
                spans = sample_results[0].tag_spans
                report_lines.append(f"\nTag Spans Found: {len(spans)}")
                for span in spans:
                    report_lines.append(
                        f"  - <{span.tag_type}>: tokens [{span.start_token}:{span.end_token}]"
                    )
            
            # 每层的跨区域注意力统计
            report_lines.append("\nCross-Region Attention Summary:")
            for r in sample_results:
                report_lines.append(f"\n  Layer {r.layer_id}:")
                
                # 找出最强的注意力流向
                max_attn = 0
                max_flow = ""
                for src, targets in r.cross_region_attention.items():
                    for tgt, attn in targets.items():
                        if attn > max_attn:
                            max_attn = attn
                            max_flow = f"{src} -> {tgt}"
                
                report_lines.append(f"    Strongest flow: {max_flow} ({max_attn:.4f})")
                
                # 各区域自注意力
                for region in ['think', 'tool_call', 'tool_response', 'answer']:
                    if region in r.cross_region_attention:
                        self_attn = r.cross_region_attention[region].get(region, 0)
                        report_lines.append(f"    {region} self-attention: {self_attn:.4f}")
        
        report = "\n".join(report_lines)
        return report


def main():
    """主函数示例"""
    import argparse
    
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
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析层索引
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
    
    # 为每个样本的每一层生成热力图
    for r in results:
        output_path = os.path.join(
            args.output_dir, 
            f"attention_heatmap_sample{r.sample_id}_layer{r.layer_id}.png"
        )
        analyzer.visualize_attention_heatmap(r, output_path=output_path)
    
    # 生成跨区域注意力对比图
    by_sample = defaultdict(list)
    for r in results:
        by_sample[r.sample_id].append(r)
    
    for sample_id, sample_results in by_sample.items():
        output_path = os.path.join(
            args.output_dir,
            f"cross_region_attention_sample{sample_id}.png"
        )
        analyzer.visualize_cross_region_attention(sample_results, output_path=output_path)
    
    # 生成报告
    report = analyzer.generate_summary_report(results)
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()