"""
Agent 轨迹频谱分析框架 (Spectral Analysis Framework for Agents)
基于 WebSailor-3B 模型的真实 KV Cache 分析

作者: Spectral Analysis Lab
日期: 2024
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, fft, fftfreq
from scipy.signal import welch
from typing import List, Dict, Tuple, Optional, Union
import re
from dataclasses import dataclass, field
from enum import Enum
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SegmentType(Enum):
    THINK = "think"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM = "system"
    USER = "user"
    ANSWER = "answer"
    OTHER = "other"


@dataclass
class Segment:
    """表示一个轨迹片段"""
    type: SegmentType
    content: str
    start_token_idx: int
    end_token_idx: int
    token_ids: List[int] = field(default_factory=list)
    
    @property
    def token_count(self) -> int:
        return self.end_token_idx - self.start_token_idx


@dataclass
class KVCacheData:
    """KV Cache 数据结构"""
    keys: torch.Tensor  # [num_layers, num_heads, seq_len, head_dim]
    values: torch.Tensor  # [num_layers, num_heads, seq_len, head_dim]
    layer_idx: Optional[int] = None
    

@dataclass
class SpectralMetrics:
    """频谱分析指标"""
    spectral_entropy: float = 0.0
    high_freq_ratio: float = 0.0
    low_freq_ratio: float = 0.0
    energy_concentration: float = 0.0
    dominant_frequency: float = 0.0
    total_energy: float = 0.0
    psd: np.ndarray = field(default_factory=lambda: np.array([]))
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))


class WebSailorKVExtractor:
    """WebSailor-3B KV Cache 提取器"""
    
    def __init__(
        self, 
        model_path: str = "/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        初始化 WebSailor-3B 模型
        
        Args:
            model_path: 模型路径
            device: 设备设置
            torch_dtype: 数据类型
            load_in_8bit: 是否使用 8bit 量化
            load_in_4bit: 是否使用 4bit 量化
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"正在加载模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
            "device_map": device,
            "torch_dtype": torch_dtype,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()
        
        # 获取模型配置
        self.config = self.model.config
        self.num_layers = getattr(self.config, 'num_hidden_layers', 64)
        self.num_heads = getattr(self.config, 'num_key_value_heads', 8)
        self.head_dim = getattr(self.config, 'hidden_size', 5120) // getattr(self.config, 'num_attention_heads', 40)
        
        print(f"模型加载完成!")
        print(f"  - 层数: {self.num_layers}")
        print(f"  - KV Heads: {self.num_heads}")
        print(f"  - Head Dim: {self.head_dim}")
    
    def tokenize_text(self, text: str) -> Dict:
        """对文本进行分词"""
        return self.tokenizer(
            text, 
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=32768
        )
    
    def extract_kv_cache(
        self, 
        text: str,
        target_layers: Optional[List[int]] = None
    ) -> Tuple[Dict[int, KVCacheData], List[int]]:
        """
        提取指定文本的 KV Cache
        
        Args:
            text: 输入文本
            target_layers: 目标层索引列表，None 表示所有层
            
        Returns:
            kv_cache_dict: {layer_idx: KVCacheData}
            token_ids: token ID 列表
        """
        if target_layers is None:
            # 默认提取中后层（更具语义信息）
            target_layers = list(range(self.num_layers // 2, self.num_layers))
        
        # 分词
        inputs = self.tokenize_text(text)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(self.model.device)
        
        token_ids = input_ids[0].tolist()
        
        print(f"输入序列长度: {len(token_ids)} tokens")
        
        # 前向传播，获取 KV Cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True
            )
        
        past_key_values = outputs.past_key_values
        
        # 提取目标层的 KV Cache
        kv_cache_dict = {}
        
        for layer_idx in target_layers:
            if layer_idx < len(past_key_values):
                key_states = past_key_values[layer_idx][0]  # [batch, num_heads, seq_len, head_dim]
                value_states = past_key_values[layer_idx][1]
                
                kv_cache_dict[layer_idx] = KVCacheData(
                    keys=key_states.cpu().float(),
                    values=value_states.cpu().float(),
                    layer_idx=layer_idx
                )
        
        # 清理 GPU 内存
        del outputs, past_key_values
        torch.cuda.empty_cache()
        gc.collect()
        
        return kv_cache_dict, token_ids
    
    def get_token_texts(self, token_ids: List[int]) -> List[str]:
        """将 token IDs 转换为文本"""
        return [self.tokenizer.decode([tid]) for tid in token_ids]


class AgentTrajectoryParser:
    """Agent 轨迹解析器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.patterns = {
            'think': (r'<think>', r'</think>'),
            'tool_call': (r'<tool_call>', r'</tool_call>'),
            'tool_response': (r'<tool_response>', r'</tool_response>'),
            'answer': (r'<answer>', r'</answer>')
        }
    
    def parse_trajectory(self, json_data: dict) -> Tuple[str, List[Segment]]:
        """
        解析 Agent 轨迹，返回完整文本和片段列表
        """
        messages = json_data.get('messages', [])
        
        # 构建完整的对话文本
        full_text = ""
        segments = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                start_idx = len(full_text)
                full_text += f"<|system|>\n{content}\n"
                # 对这部分进行分词以获取 token 边界
                segment_tokens = self.tokenizer.encode(f"<|system|>\n{content}\n", add_special_tokens=False)
                segments.append(Segment(
                    type=SegmentType.SYSTEM,
                    content=content,
                    start_token_idx=0,  # 稍后更新
                    end_token_idx=0,
                    token_ids=segment_tokens
                ))
                
            elif role == 'user':
                start_idx = len(full_text)
                if '<tool_response>' in content:
                    full_text += f"<|user|>\n{content}\n"
                    segment_tokens = self.tokenizer.encode(f"<|user|>\n{content}\n", add_special_tokens=False)
                    segments.append(Segment(
                        type=SegmentType.TOOL_RESPONSE,
                        content=content,
                        start_token_idx=0,
                        end_token_idx=0,
                        token_ids=segment_tokens
                    ))
                else:
                    full_text += f"<|user|>\n{content}\n"
                    segment_tokens = self.tokenizer.encode(f"<|user|>\n{content}\n", add_special_tokens=False)
                    segments.append(Segment(
                        type=SegmentType.USER,
                        content=content,
                        start_token_idx=0,
                        end_token_idx=0,
                        token_ids=segment_tokens
                    ))
                    
            elif role == 'assistant':
                # 解析 assistant 消息中的各个部分
                parsed_segments = self._parse_assistant_content(content)
                full_text += f"<|assistant|>\n{content}\n"
                segments.extend(parsed_segments)
        
        # 更新 token 索引
        self._update_token_indices(full_text, segments)
        
        return full_text, segments
    
    def _parse_assistant_content(self, content: str) -> List[Segment]:
        """解析 assistant 消息内容"""
        segments = []
        
        # 解析 <think>...</think>
        think_pattern = r'<think>(.*?)</think>'
        for match in re.finditer(think_pattern, content, re.DOTALL):
            think_content = match.group(1)
            segment_tokens = self.tokenizer.encode(f"<think>{think_content}</think>", add_special_tokens=False)
            segments.append(Segment(
                type=SegmentType.THINK,
                content=think_content,
                start_token_idx=0,
                end_token_idx=0,
                token_ids=segment_tokens
            ))
        
        # 解析 <tool_call>...</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        for match in re.finditer(tool_call_pattern, content, re.DOTALL):
            call_content = match.group(1)
            segment_tokens = self.tokenizer.encode(f"<tool_call>{call_content}</tool_call>", add_special_tokens=False)
            segments.append(Segment(
                type=SegmentType.TOOL_CALL,
                content=call_content,
                start_token_idx=0,
                end_token_idx=0,
                token_ids=segment_tokens
            ))
        
        # 解析 <answer>...</answer>
        answer_pattern = r'<answer>(.*?)</answer>'
        for match in re.finditer(answer_pattern, content, re.DOTALL):
            answer_content = match.group(1)
            segment_tokens = self.tokenizer.encode(f"<answer>{answer_content}</answer>", add_special_tokens=False)
            segments.append(Segment(
                type=SegmentType.ANSWER,
                content=answer_content,
                start_token_idx=0,
                end_token_idx=0,
                token_ids=segment_tokens
            ))
        
        return segments
    
    def _update_token_indices(self, full_text: str, segments: List[Segment]):
        """更新各片段的 token 索引"""
        # 对完整文本进行分词
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        current_idx = 0
        for seg in segments:
            seg.start_token_idx = current_idx
            seg.end_token_idx = current_idx + len(seg.token_ids)
            current_idx = seg.end_token_idx


class KVCacheSpectralAnalyzer:
    """KV Cache 频谱分析器"""
    
    def __init__(self, cutoff_ratio: float = 0.2):
        """
        初始化频谱分析器
        
        Args:
            cutoff_ratio: 高频截止比例（用于计算 HFR）
        """
        self.cutoff_ratio = cutoff_ratio
    
    def extract_segment_kv(
        self, 
        kv_cache: KVCacheData, 
        segment: Segment
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 KV Cache 中提取指定片段的数据
        
        Args:
            kv_cache: 完整的 KV Cache
            segment: 目标片段
            
        Returns:
            keys: [seq_len, num_heads * head_dim]
            values: [seq_len, num_heads * head_dim]
        """
        start_idx = segment.start_token_idx
        end_idx = segment.end_token_idx
        
        # 提取对应位置的 KV
        # 原始形状: [batch, num_heads, seq_len, head_dim]
        keys = kv_cache.keys[0, :, start_idx:end_idx, :].numpy()  # [num_heads, seg_len, head_dim]
        values = kv_cache.values[0, :, start_idx:end_idx, :].numpy()
        
        # 重塑为 [seg_len, num_heads * head_dim]
        num_heads, seg_len, head_dim = keys.shape
        keys = keys.transpose(1, 0, 2).reshape(seg_len, -1)
        values = values.transpose(1, 0, 2).reshape(seg_len, -1)
        
        return keys, values
    
    def compute_dct_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """计算 DCT 频谱"""
        if len(signal) < 2:
            return np.array([1.0])
        return dct(signal, type=2, norm='ortho')
    
    def compute_power_spectral_density(self, signal: np.ndarray) -> np.ndarray:
        """计算功率谱密度"""
        spectrum = self.compute_dct_spectrum(signal)
        return np.abs(spectrum) ** 2
    
    def compute_spectral_entropy(self, psd: np.ndarray) -> float:
        """
        计算谱熵 (Spectral Entropy)
        
        SE = -Σ p(f) * log2(p(f))
        """
        psd = psd + 1e-10  # 避免 log(0)
        p = psd / np.sum(psd)
        entropy = -np.sum(p * np.log2(p))
        
        # 归一化
        max_entropy = np.log2(len(psd)) if len(psd) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def compute_high_freq_ratio(self, psd: np.ndarray) -> float:
        """
        计算高频能量占比 (High-Frequency Ratio)
        
        HFR = Σ P(f > fc) / Σ P(f)
        """
        if len(psd) < 2:
            return 0
        cutoff = int(len(psd) * (1 - self.cutoff_ratio))
        high_freq_energy = np.sum(psd[cutoff:])
        total_energy = np.sum(psd)
        return high_freq_energy / (total_energy + 1e-10)
    
    def compute_low_freq_ratio(self, psd: np.ndarray) -> float:
        """计算低频能量占比"""
        if len(psd) < 2:
            return 1.0
        cutoff = int(len(psd) * self.cutoff_ratio)
        low_freq_energy = np.sum(psd[:cutoff])
        total_energy = np.sum(psd)
        return low_freq_energy / (total_energy + 1e-10)
    
    def compute_energy_concentration(self, psd: np.ndarray, top_k: int = 5) -> float:
        """计算能量集中度（前 k 个频率分量的能量占比）"""
        if len(psd) < top_k:
            return 1.0
        sorted_psd = np.sort(psd)[::-1]
        top_energy = np.sum(sorted_psd[:top_k])
        total_energy = np.sum(psd)
        return top_energy / (total_energy + 1e-10)
    
    def analyze_segment_kv(
        self, 
        keys: np.ndarray, 
        values: np.ndarray,
        use_keys: bool = True
    ) -> SpectralMetrics:
        """
        分析片段的 KV Cache 频谱特性
        
        Args:
            keys: [seq_len, dim]
            values: [seq_len, dim]
            use_keys: 是否使用 keys（否则使用 values）
            
        Returns:
            SpectralMetrics
        """
        data = keys if use_keys else values
        
        if data.shape[0] < 4:
            return SpectralMetrics()
        
        seq_len, dim = data.shape
        
        # 对每个维度计算 PSD，然后平均
        all_psd = []
        for d in range(dim):
            signal = data[:, d]
            # 去均值
            signal = signal - np.mean(signal)
            if np.std(signal) > 1e-8:
                signal = signal / np.std(signal)
            
            psd = self.compute_power_spectral_density(signal)
            all_psd.append(psd)
        
        # 平均 PSD
        avg_psd = np.mean(all_psd, axis=0)
        frequencies = np.arange(len(avg_psd)) / len(avg_psd)
        
        return SpectralMetrics(
            spectral_entropy=self.compute_spectral_entropy(avg_psd),
            high_freq_ratio=self.compute_high_freq_ratio(avg_psd),
            low_freq_ratio=self.compute_low_freq_ratio(avg_psd),
            energy_concentration=self.compute_energy_concentration(avg_psd),
            dominant_frequency=frequencies[np.argmax(avg_psd)] if len(avg_psd) > 0 else 0,
            total_energy=np.sum(avg_psd),
            psd=avg_psd,
            frequencies=frequencies
        )
    
    def compute_spectral_stability(
        self, 
        psd1: np.ndarray, 
        psd2: np.ndarray
    ) -> float:
        """
        计算两个片段之间的频谱稳定性（余弦相似度）
        """
        # 对齐长度
        min_len = min(len(psd1), len(psd2))
        psd1 = psd1[:min_len]
        psd2 = psd2[:min_len]
        
        # 归一化
        psd1 = psd1 / (np.linalg.norm(psd1) + 1e-10)
        psd2 = psd2 / (np.linalg.norm(psd2) + 1e-10)
        
        return np.dot(psd1, psd2)


class KVCacheVisualizer:
    """KV Cache 可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
        self.colors = {
            SegmentType.THINK: '#FF6B6B',
            SegmentType.TOOL_CALL: '#4ECDC4',
            SegmentType.TOOL_RESPONSE: '#45B7D1',
            SegmentType.SYSTEM: '#95A5A6',
            SegmentType.USER: '#F39C12',
            SegmentType.ANSWER: '#9B59B6',
            SegmentType.OTHER: '#BDC3C7'
        }
    
    def plot_segment_statistics(
        self, 
        segments: List[Segment],
        metrics_dict: Dict[SegmentType, List[SpectralMetrics]],
        save_path: Optional[str] = None
    ):
        """绘制片段统计和频谱指标"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 片段数量统计
        type_counts = {}
        type_tokens = {}
        for seg in segments:
            t = seg.type
            type_counts[t] = type_counts.get(t, 0) + 1
            type_tokens[t] = type_tokens.get(t, 0) + seg.token_count
        
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        colors = [self.colors.get(t, '#333') for t in types]
        
        axes[0, 0].bar([t.value for t in types], counts, color=colors)
        axes[0, 0].set_title('Segment Count by Type', fontsize=12)
        axes[0, 0].set_xlabel('Segment Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Token 数量统计
        tokens = [type_tokens[t] for t in types]
        axes[0, 1].bar([t.value for t in types], tokens, color=colors)
        axes[0, 1].set_title('Token Count by Type', fontsize=12)
        axes[0, 1].set_xlabel('Segment Type')
        axes[0, 1].set_ylabel('Token Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 谱熵分布
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        entropy_data = []
        labels = []
        box_colors = []
        
        for t in target_types:
            if t in metrics_dict and metrics_dict[t]:
                entropy_data.append([m.spectral_entropy for m in metrics_dict[t]])
                labels.append(t.value)
                box_colors.append(self.colors[t])
        
        if entropy_data:
            bp = axes[1, 0].boxplot(entropy_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[1, 0].set_title('Spectral Entropy Distribution', fontsize=12)
        axes[1, 0].set_ylabel('Spectral Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 高频能量占比
        hfr_data = []
        labels = []
        bar_colors = []
        
        for t in target_types:
            if t in metrics_dict and metrics_dict[t]:
                hfr_data.append(np.mean([m.high_freq_ratio for m in metrics_dict[t]]))
                labels.append(t.value)
                bar_colors.append(self.colors[t])
        
        if hfr_data:
            axes[1, 1].bar(labels, hfr_data, color=bar_colors, alpha=0.7)
        axes[1, 1].set_title('Average High-Frequency Ratio', fontsize=12)
        axes[1, 1].set_ylabel('HFR')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存: {save_path}")
        plt.show()
    
    def plot_spectrogram(
        self,
        kv_cache: KVCacheData,
        segments: List[Segment],
        save_path: Optional[str] = None
    ):
        """绘制 KV Cache 频谱热力图"""
        # 提取 Keys 数据
        keys = kv_cache.keys[0].numpy()  # [num_heads, seq_len, head_dim]
        num_heads, seq_len, head_dim = keys.shape
        
        # 平均所有 heads
        keys_avg = np.mean(keys, axis=0)  # [seq_len, head_dim]
        
        # 计算滑动窗口频谱
        window_size = min(64, seq_len // 4)
        if window_size < 4:
            window_size = 4
        hop_size = max(1, window_size // 2)
        
        spectrograms = []
        positions = []
        
        for i in range(0, seq_len - window_size, hop_size):
            window = keys_avg[i:i+window_size]
            # 对所有维度平均
            avg_signal = np.mean(window, axis=1)
            avg_signal = avg_signal - np.mean(avg_signal)
            if np.std(avg_signal) > 1e-8:
                avg_signal = avg_signal / np.std(avg_signal)
            spectrum = np.abs(dct(avg_signal, norm='ortho'))
            spectrograms.append(spectrum)
            positions.append(i + window_size // 2)
        
        if not spectrograms:
            print("数据不足，无法生成频谱图")
            return
        
        spectrogram = np.array(spectrograms).T
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        im = ax.imshow(
            np.log1p(spectrogram), 
            aspect='auto', 
            cmap='viridis',
            extent=[0, seq_len, window_size//2, 0],
            interpolation='bilinear'
        )
        
        # 标注片段边界
        for seg in segments:
            if seg.type in [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]:
                color = self.colors[seg.type]
                ax.axvline(x=seg.start_token_idx, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        
        plt.colorbar(im, ax=ax, label='Log Power')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Frequency (DCT index)', fontsize=12)
        ax.set_title(f'KV Cache Spectrogram (Layer {kv_cache.layer_idx})\n'
                    f'Red: Think, Cyan: Tool Call, Blue: Tool Response', fontsize=14)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors[SegmentType.THINK], linestyle='--', label='Think'),
            plt.Line2D([0], [0], color=self.colors[SegmentType.TOOL_CALL], linestyle='--', label='Tool Call'),
            plt.Line2D([0], [0], color=self.colors[SegmentType.TOOL_RESPONSE], linestyle='--', label='Tool Response'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存: {save_path}")
        plt.show()
    
    def plot_energy_decay_curves(
        self,
        metrics_dict: Dict[SegmentType, List[SpectralMetrics]],
        save_path: Optional[str] = None
    ):
        """绘制能量衰减曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        
        for seg_type in target_types:
            if seg_type not in metrics_dict or not metrics_dict[seg_type]:
                continue
            
            all_psd = []
            for m in metrics_dict[seg_type]:
                if len(m.psd) > 1:
                    # 插值到统一长度
                    target_len = 64
                    x_old = np.linspace(0, 1, len(m.psd))
                    x_new = np.linspace(0, 1, target_len)
                    psd_interp = np.interp(x_new, x_old, m.psd)
                    # 归一化
                    psd_interp = psd_interp / (np.max(psd_interp) + 1e-10)
                    all_psd.append(psd_interp)
            
            if all_psd:
                mean_psd = np.mean(all_psd, axis=0)
                std_psd = np.std(all_psd, axis=0)
                frequencies = np.linspace(0, 1, len(mean_psd))
                
                color = self.colors[seg_type]
                ax.plot(frequencies, mean_psd, color=color, label=seg_type.value, linewidth=2)
                ax.fill_between(frequencies, mean_psd - std_psd, mean_psd + std_psd,
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Normalized Frequency', fontsize=12)
        ax.set_ylabel('Normalized Power Spectral Density', fontsize=12)
        ax.set_title('Energy Decay Curves by Segment Type', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xlim([0, 0.5])
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存: {save_path}")
        plt.show()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[SegmentType, List[SpectralMetrics]],
        save_path: Optional[str] = None
    ):
        """绘制频谱指标对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        metric_names = ['Spectral Entropy', 'High-Freq Ratio', 'Energy Concentration']
        
        for idx, metric_name in enumerate(metric_names):
            data = []
            labels = []
            colors = []
            
            for seg_type in target_types:
                if seg_type in metrics_dict and metrics_dict[seg_type]:
                    if metric_name == 'Spectral Entropy':
                        values = [m.spectral_entropy for m in metrics_dict[seg_type]]
                    elif metric_name == 'High-Freq Ratio':
                        values = [m.high_freq_ratio for m in metrics_dict[seg_type]]
                    else:
                        values = [m.energy_concentration for m in metrics_dict[seg_type]]
                    
                    data.append(values)
                    labels.append(seg_type.value)
                    colors.append(self.colors[seg_type])
            
            if data:
                bp = axes[idx].boxplot(data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            axes[idx].set_title(metric_name, fontsize=12)
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=30)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Spectral Metrics Comparison (from KV Cache)', fontsize=14, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存: {save_path}")
        plt.show()


class AgentTrajectoryExperiment:
    """Agent 轨迹频谱分析实验"""
    
    def __init__(
        self,
        model_path: str = "/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B",
        target_layers: Optional[List[int]] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False
    ):
        """
        初始化实验
        
        Args:
            model_path: 模型路径
            target_layers: 目标分析层
            device: 设备
            torch_dtype: 数据类型
            load_in_4bit: 是否使用 4bit 量化
        """
        print("=" * 60)
        print("Agent 轨迹频谱分析实验")
        print("基于 WebSailor-3B 真实 KV Cache")
        print("=" * 60)
        
        # 初始化 KV 提取器
        self.kv_extractor = WebSailorKVExtractor(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit
        )
        
        # 初始化解析器
        self.parser = AgentTrajectoryParser(self.kv_extractor.tokenizer)
        
        # 初始化分析器
        self.spectral_analyzer = KVCacheSpectralAnalyzer(cutoff_ratio=0.2)
        
        # 初始化可视化器
        self.visualizer = KVCacheVisualizer()
        
        # 设置目标层
        if target_layers is None:
            # 默认分析中后层
            num_layers = self.kv_extractor.num_layers
            self.target_layers = [num_layers // 2, num_layers * 3 // 4, num_layers - 1]
        else:
            self.target_layers = target_layers
        
        print(f"目标分析层: {self.target_layers}")
    
    def run_analysis(self, json_data: dict) -> Dict:
        """
        运行完整的频谱分析
        
        Args:
            json_data: Agent 轨迹 JSON 数据
            
        Returns:
            分析结果字典
        """
        print("\n" + "=" * 60)
        print("开始分析...")
        print("=" * 60)
        
        # 1. 解析轨迹
        print("\n[1/4] 解析轨迹...")
        full_text, segments = self.parser.parse_trajectory(json_data)
        
        print(f"  - 完整文本长度: {len(full_text)} 字符")
        print(f"  - 片段数量: {len(segments)}")
        
        # 统计各类型
        type_counts = {}
        for seg in segments:
            type_counts[seg.type.value] = type_counts.get(seg.type.value, 0) + 1
        for t, c in type_counts.items():
            print(f"    - {t}: {c}")
        
        # 2. 提取 KV Cache
        print("\n[2/4] 提取 KV Cache...")
        kv_cache_dict, token_ids = self.kv_extractor.extract_kv_cache(
            full_text, 
            target_layers=self.target_layers
        )
        
        print(f"  - 提取了 {len(kv_cache_dict)} 层的 KV Cache")
        print(f"  - Token 序列长度: {len(token_ids)}")
        
        # 3. 分析各片段的频谱特性
        print("\n[3/4] 分析频谱特性...")
        
        # 选择一个层进行主要分析
        main_layer = self.target_layers[len(self.target_layers) // 2]
        main_kv = kv_cache_dict[main_layer]
        
        metrics_by_type: Dict[SegmentType, List[SpectralMetrics]] = {
            t: [] for t in SegmentType
        }
        
        segment_metrics = []
        
        for seg in tqdm(segments, desc="分析片段"):
            if seg.token_count < 4:
                continue
            
            # 确保索引不越界
            if seg.end_token_idx > main_kv.keys.shape[2]:
                continue
            
            keys, values = self.spectral_analyzer.extract_segment_kv(main_kv, seg)
            metrics = self.spectral_analyzer.analyze_segment_kv(keys, values, use_keys=True)
            
            metrics_by_type[seg.type].append(metrics)
            segment_metrics.append({
                'segment': seg,
                'metrics': metrics
            })
        
        # 4. 汇总结果
        print("\n[4/4] 汇总结果...")
        
        summary = {}
        for seg_type, metrics_list in metrics_by_type.items():
            if metrics_list:
                summary[seg_type.value] = {
                    'count': len(metrics_list),
                    'spectral_entropy_mean': np.mean([m.spectral_entropy for m in metrics_list]),
                    'spectral_entropy_std': np.std([m.spectral_entropy for m in metrics_list]),
                    'high_freq_ratio_mean': np.mean([m.high_freq_ratio for m in metrics_list]),
                    'high_freq_ratio_std': np.std([m.high_freq_ratio for m in metrics_list]),
                    'low_freq_ratio_mean': np.mean([m.low_freq_ratio for m in metrics_list]),
                    'energy_concentration_mean': np.mean([m.energy_concentration for m in metrics_list]),
                    'energy_concentration_std': np.std([m.energy_concentration for m in metrics_list]),
                }
        
        results = {
            'segments': segments,
            'kv_cache': main_kv,
            'metrics_by_type': metrics_by_type,
            'segment_metrics': segment_metrics,
            'summary': summary,
            'layer_idx': main_layer
        }
        
        return results
    
    def print_report(self, results: Dict):
        """打印分析报告"""
        print("\n" + "=" * 60)
        print("频谱分析报告")
        print("=" * 60)
        
        summary = results['summary']
        
        print(f"\n分析层: Layer {results['layer_idx']}")
        print(f"总片段数: {len(results['segments'])}")
        
        print("\n" + "-" * 40)
        print("各类型片段的频谱特性:")
        print("-" * 40)
        
        for seg_type, stats in summary.items():
            print(f"\n[{seg_type}] ({stats['count']} 个片段)")
            print(f"  谱熵 (SE):        {stats['spectral_entropy_mean']:.4f} ± {stats['spectral_entropy_std']:.4f}")
            print(f"  高频能量占比 (HFR): {stats['high_freq_ratio_mean']:.4f} ± {stats['high_freq_ratio_std']:.4f}")
            print(f"  低频能量占比:      {stats['low_freq_ratio_mean']:.4f}")
            print(f"  能量集中度:        {stats['energy_concentration_mean']:.4f} ± {stats['energy_concentration_std']:.4f}")
        
        # 验证假设
        print("\n" + "-" * 40)
        print("假设验证:")
        print("-" * 40)
        
        if 'think' in summary and 'tool_call' in summary:
            think_se = summary['think']['spectral_entropy_mean']
            call_se = summary['tool_call']['spectral_entropy_mean']
            
            if think_se > call_se:
                print(f"✓ Think 的谱熵 ({think_se:.4f}) > Tool Call ({call_se:.4f})")
                print("  → 支持假设: Think 信号更接近噪声，Tool Call 更结构化")
            else:
                print(f"✗ Think 的谱熵 ({think_se:.4f}) ≤ Tool Call ({call_se:.4f})")
            
            think_hfr = summary['think']['high_freq_ratio_mean']
            call_hfr = summary['tool_call']['high_freq_ratio_mean']
            
            if think_hfr > call_hfr:
                print(f"✓ Think 的 HFR ({think_hfr:.4f}) > Tool Call ({call_hfr:.4f})")
                print("  → 支持假设: Think 包含更多高频噪声，可进行低通滤波压缩")
            else:
                print(f"✗ Think 的 HFR ({think_hfr:.4f}) ≤ Tool Call ({call_hfr:.4f})")
    
    def visualize_results(self, results: Dict, save_dir: str = "./"):
        """生成可视化图表"""
        print("\n生成可视化图表...")
        
        # 1. 片段统计和基本指标
        self.visualizer.plot_segment_statistics(
            results['segments'],
            results['metrics_by_type'],
            save_path=f"{save_dir}/segment_statistics.png"
        )
        
        # 2. 频谱热力图
        self.visualizer.plot_spectrogram(
            results['kv_cache'],
            results['segments'],
            save_path=f"{save_dir}/kv_spectrogram.png"
        )
        
        # 3. 能量衰减曲线
        self.visualizer.plot_energy_decay_curves(
            results['metrics_by_type'],
            save_path=f"{save_dir}/energy_decay.png"
        )
        
        # 4. 指标对比
        self.visualizer.plot_metrics_comparison(
            results['metrics_by_type'],
            save_path=f"{save_dir}/metrics_comparison.png"
        )


def load_json_file(filepath: str) -> dict:
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数"""
    # 示例数据
    sample_data = {
        "question": "I'm researching species that became invasive after people who kept them as pets released them...",
        "answer": "34689",
        "rollout_id": 1,
        "messages": [
            {
                "role": "system", 
                "content": "You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions."
            },
            {
                "role": "user", 
                "content": "A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling tools.\n\nUser: I'm researching species that became invasive after people who kept them as pets released them. There's a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020?"
            },
            {
                "role": "assistant", 
                "content": """<think>To determine the exact zip codes where the fish species from Finding Nemo became established as nonnative, I need highly specific, authoritative information, ideally from the USGS or similar wildlife agencies. The most effective way to pinpoint this is to use targeted search queries that combine the movie character, the species (likely "Amphiprion ocellaris," the clownfish), and terms like "nonnative," "found," and "zip codes." By including both the movie context and the USGS angle, I maximize the chances of finding official reports or research summaries that list the zip codes where this species was first introduced or found as nonnative. This approach should yield the precise, geographic data required for the answer. I need to run these searches now to locate the relevant zip codes.
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["Finding Nemo fish nonnative found zip codes", "USGS Finding Nemo fish nonnative USGS", "Finding Nemo fish USGS found before 2020 zip codes"]}}
</tool_call>"""
            },
            {
                "role": "user", 
                "content": """<tool_response>
A Google search for 'Finding Nemo fish nonnative found zip codes' found 9 results:

## Web Results
1. [clown anemonefish (Amphiprion ocellaris) - Species Profile](https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID=3243)
One individual identified as Amphiprion ocellaris was collected in July 2018 in Fred Howard Park, Pinellas Co., Florida

Table 1. States with nonindigenous occurrences:
State: FL, First Observed: 2018, Last Observed: 2018, Total HUCs: 1, HUC: Crystal-Pithlachascotee

Status: Eradicated. Specimen was removed alive and transferred to the Clearwater Marine Aquarium.
</tool_response>"""
            },
            {
                "role": "assistant", 
                "content": """<think>Based on the USGS Nonindigenous Aquatic Species database, I found that the clown anemonefish (Amphiprion ocellaris), the species popularized by Finding Nemo, was collected as a nonnative species in July 2018 at Fred Howard Park in Pinellas County, Florida. The specimen was found in the Crystal-Pithlachascotee drainage area and was subsequently removed and transferred to the Clearwater Marine Aquarium.

Now I need to find the zip code for Fred Howard Park in Pinellas County, Florida. Fred Howard Park is located in Tarpon Springs, Florida. The zip code for this area is 34689.

The USGS record shows this was the only documented nonnative occurrence before 2020.
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["Fred Howard Park Tarpon Springs zip code", "Fred Howard Park Pinellas County Florida zip code"]}}
</tool_call>"""
            },
            {
                "role": "user",
                "content": """<tool_response>
A Google search for 'Fred Howard Park Tarpon Springs zip code' found results:

Fred Howard Park is located at 1700 Sunset Dr, Tarpon Springs, FL 34689

The park is a 155-acre Pinellas County park located in Tarpon Springs, Florida.
</tool_response>"""
            },
            {
                "role": "assistant",
                "content": """<think>I have confirmed all the necessary information:

1. The fish species from Finding Nemo is the clown anemonefish (Amphiprion ocellaris)
2. According to the USGS Nonindigenous Aquatic Species database, this species was found as a nonnative species at Fred Howard Park, Pinellas County, Florida in July 2018
3. This was before the year 2020
4. Fred Howard Park is located in Tarpon Springs, FL with zip code 34689

The answer is the five-digit zip code: 34689
</think>
<answer>34689</answer>"""
            }
        ],
        "prediction": "34689",
        "termination": "completed",
        "tool_call_search_times": 2,
        "tool_call_visit_times": 0
    }
    
    # 创建实验
    experiment = AgentTrajectoryExperiment(
        model_path="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B",
        target_layers=None,  # 自动选择中后层
        device="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=False  # 使用 4bit 量化以节省显存
    )
    
    # 运行分析
    results = experiment.run_analysis(sample_data)
    
    # 打印报告
    experiment.print_report(results)
    
    # 生成可视化
    experiment.visualize_results(results, save_dir="./spectral_analysis_results")
    
    print("\n分析完成!")
    
    return results


if __name__ == "__main__":
    results = main()