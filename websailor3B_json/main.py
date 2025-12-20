"""
Agent 轨迹频谱分析框架 (Spectral Analysis Framework for Agents)
基于 WebSailor-3B 模型的真实 KV Cache 分析

支持批量处理多个 Agent 轨迹
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass, field
from enum import Enum
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc
import os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

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
    """轨迹片段"""
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
    """KV Cache 数据"""
    keys: torch.Tensor
    values: torch.Tensor
    layer_idx: int = 0


@dataclass
class SpectralMetrics:
    """频谱指标"""
    spectral_entropy: float = 0.0
    high_freq_ratio: float = 0.0
    low_freq_ratio: float = 0.0
    energy_concentration: float = 0.0
    dominant_frequency: float = 0.0
    total_energy: float = 0.0
    psd: np.ndarray = field(default_factory=lambda: np.array([]))
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TrajectoryResult:
    """单条轨迹分析结果"""
    trajectory_id: int
    question: str
    answer: str
    prediction: str
    is_correct: bool
    segments: List[Segment]
    metrics_by_type: Dict[SegmentType, List[SpectralMetrics]]
    summary: Dict
    token_count: int
    layer_idx: int


class WebSailorKVExtractor:
    """WebSailor KV Cache 提取器"""
    
    def __init__(
        self, 
        model_path: str = "/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"加载模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
            "device_map": device,
            "torch_dtype": torch_dtype,
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()
        
        self.config = self.model.config
        self.num_layers = getattr(self.config, 'num_hidden_layers', 36)
        self.num_heads = getattr(self.config, 'num_key_value_heads', 4)
        self.head_dim = getattr(self.config, 'hidden_size', 3072) // getattr(self.config, 'num_attention_heads', 24)
        
        print(f"模型加载完成: {self.num_layers} 层, {self.num_heads} KV heads")
    
    def extract_kv_cache(
        self, 
        text: str,
        target_layers: Optional[List[int]] = None,
        max_length: int = 8192
    ) -> Tuple[Dict[int, KVCacheData], List[int]]:
        """提取 KV Cache"""
        if target_layers is None:
            target_layers = [self.num_layers // 2, self.num_layers * 3 // 4, self.num_layers - 1]
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(self.model.device)
        
        token_ids = input_ids[0].tolist()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        
        past_key_values = outputs.past_key_values
        
        kv_cache_dict = {}
        for layer_idx in target_layers:
            if layer_idx < len(past_key_values):
                kv_cache_dict[layer_idx] = KVCacheData(
                    keys=past_key_values[layer_idx][0].cpu().float(),
                    values=past_key_values[layer_idx][1].cpu().float(),
                    layer_idx=layer_idx
                )
        
        del outputs, past_key_values
        torch.cuda.empty_cache()
        gc.collect()
        
        return kv_cache_dict, token_ids


class AgentTrajectoryParser:
    """轨迹解析器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def parse_trajectory(self, json_data: dict) -> Tuple[str, List[Segment]]:
        """解析轨迹"""
        messages = json_data.get('messages', [])
        
        full_text = ""
        segments = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                segment_text = f"<|system|>\n{content}\n"
                segment_tokens = self.tokenizer.encode(segment_text, add_special_tokens=False)
                segments.append(Segment(
                    type=SegmentType.SYSTEM,
                    content=content,
                    start_token_idx=0,
                    end_token_idx=0,
                    token_ids=segment_tokens
                ))
                full_text += segment_text
                
            elif role == 'user':
                segment_text = f"<|user|>\n{content}\n"
                segment_tokens = self.tokenizer.encode(segment_text, add_special_tokens=False)
                seg_type = SegmentType.TOOL_RESPONSE if '<tool_response>' in content else SegmentType.USER
                segments.append(Segment(
                    type=seg_type,
                    content=content,
                    start_token_idx=0,
                    end_token_idx=0,
                    token_ids=segment_tokens
                ))
                full_text += segment_text
                    
            elif role == 'assistant':
                full_text += f"<|assistant|>\n{content}\n"
                segments.extend(self._parse_assistant_content(content))
        
        self._update_token_indices(segments)
        
        return full_text, segments
    
    def _parse_assistant_content(self, content: str) -> List[Segment]:
        """解析 assistant 内容"""
        segments = []
        patterns = [
            (r'<think>(.*?)</think>', SegmentType.THINK),
            (r'<tool_call>(.*?)</tool_call>', SegmentType.TOOL_CALL),
            (r'<answer>(.*?)</answer>', SegmentType.ANSWER),
        ]
        
        for pattern, seg_type in patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                inner_content = match.group(1)
                full_match = match.group(0)
                segment_tokens = self.tokenizer.encode(full_match, add_special_tokens=False)
                segments.append(Segment(
                    type=seg_type,
                    content=inner_content,
                    start_token_idx=0,
                    end_token_idx=0,
                    token_ids=segment_tokens
                ))
        
        return segments
    
    def _update_token_indices(self, segments: List[Segment]):
        """更新 token 索引"""
        current_idx = 0
        for seg in segments:
            seg.start_token_idx = current_idx
            seg.end_token_idx = current_idx + len(seg.token_ids)
            current_idx = seg.end_token_idx


class KVCacheSpectralAnalyzer:
    """频谱分析器"""
    
    def __init__(self, cutoff_ratio: float = 0.2):
        self.cutoff_ratio = cutoff_ratio
    
    def extract_segment_kv(self, kv_cache: KVCacheData, segment: Segment) -> Tuple[np.ndarray, np.ndarray]:
        """提取片段 KV"""
        start_idx = segment.start_token_idx
        end_idx = min(segment.end_token_idx, kv_cache.keys.shape[2])
        
        if start_idx >= end_idx:
            return np.array([]), np.array([])
        
        keys = kv_cache.keys[0, :, start_idx:end_idx, :].numpy()
        values = kv_cache.values[0, :, start_idx:end_idx, :].numpy()
        
        num_heads, seg_len, head_dim = keys.shape
        keys = keys.transpose(1, 0, 2).reshape(seg_len, -1)
        values = values.transpose(1, 0, 2).reshape(seg_len, -1)
        
        return keys, values
    
    def analyze_segment_kv(self, keys: np.ndarray, values: np.ndarray) -> SpectralMetrics:
        """分析片段频谱"""
        data = keys if keys.size > 0 else values
        
        if data.size == 0 or data.shape[0] < 4:
            return SpectralMetrics()
        
        seq_len, dim = data.shape
        
        # 采样维度加速计算
        sample_dims = min(dim, 128)
        dim_indices = np.linspace(0, dim-1, sample_dims, dtype=int)
        
        all_psd = []
        for d in dim_indices:
            signal = data[:, d].copy()
            signal = signal - np.mean(signal)
            std = np.std(signal)
            if std > 1e-8:
                signal = signal / std
            spectrum = dct(signal, type=2, norm='ortho')
            psd = np.abs(spectrum) ** 2
            all_psd.append(psd)
        
        avg_psd = np.mean(all_psd, axis=0)
        frequencies = np.arange(len(avg_psd)) / len(avg_psd)
        
        return SpectralMetrics(
            spectral_entropy=self._compute_entropy(avg_psd),
            high_freq_ratio=self._compute_hfr(avg_psd),
            low_freq_ratio=self._compute_lfr(avg_psd),
            energy_concentration=self._compute_concentration(avg_psd),
            dominant_frequency=frequencies[np.argmax(avg_psd)] if len(avg_psd) > 0 else 0,
            total_energy=np.sum(avg_psd),
            psd=avg_psd,
            frequencies=frequencies
        )
    
    def _compute_entropy(self, psd: np.ndarray) -> float:
        psd = psd + 1e-10
        p = psd / np.sum(psd)
        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(len(psd)) if len(psd) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _compute_hfr(self, psd: np.ndarray) -> float:
        if len(psd) < 2:
            return 0
        cutoff = int(len(psd) * (1 - self.cutoff_ratio))
        return np.sum(psd[cutoff:]) / (np.sum(psd) + 1e-10)
    
    def _compute_lfr(self, psd: np.ndarray) -> float:
        if len(psd) < 2:
            return 1.0
        cutoff = max(1, int(len(psd) * self.cutoff_ratio))
        return np.sum(psd[:cutoff]) / (np.sum(psd) + 1e-10)
    
    def _compute_concentration(self, psd: np.ndarray, top_k: int = 5) -> float:
        if len(psd) < top_k:
            return 1.0
        sorted_psd = np.sort(psd)[::-1]
        return np.sum(sorted_psd[:top_k]) / (np.sum(psd) + 1e-10)


class BatchVisualizer:
    """批量可视化器"""
    
    def __init__(self, save_dir: str = "./spectral_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            SegmentType.THINK: '#FF6B6B',
            SegmentType.TOOL_CALL: '#4ECDC4',
            SegmentType.TOOL_RESPONSE: '#45B7D1',
            SegmentType.SYSTEM: '#95A5A6',
            SegmentType.USER: '#F39C12',
            SegmentType.ANSWER: '#9B59B6',
            SegmentType.OTHER: '#BDC3C7'
        }
    
    def plot_aggregate_statistics(
        self, 
        all_results: List[TrajectoryResult],
        save_name: str = "aggregate_statistics.png"
    ):
        """绘制聚合统计"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 收集所有指标
        metrics_by_type = {t: {'entropy': [], 'hfr': [], 'lfr': [], 'concentration': []} 
                          for t in SegmentType}
        
        for result in all_results:
            for seg_type, metrics_list in result.metrics_by_type.items():
                for m in metrics_list:
                    metrics_by_type[seg_type]['entropy'].append(m.spectral_entropy)
                    metrics_by_type[seg_type]['hfr'].append(m.high_freq_ratio)
                    metrics_by_type[seg_type]['lfr'].append(m.low_freq_ratio)
                    metrics_by_type[seg_type]['concentration'].append(m.energy_concentration)
        
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        
        # 1. 谱熵分布
        entropy_data = []
        labels = []
        colors = []
        for t in target_types:
            if metrics_by_type[t]['entropy']:
                entropy_data.append(metrics_by_type[t]['entropy'])
                labels.append(t.value)
                colors.append(self.colors[t])
        
        if entropy_data:
            bp = axes[0, 0].boxplot(entropy_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[0, 0].set_title('Spectral Entropy Distribution', fontsize=12)
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. HFR 分布
        hfr_data = []
        for t in target_types:
            if metrics_by_type[t]['hfr']:
                hfr_data.append(metrics_by_type[t]['hfr'])
        
        if hfr_data:
            bp = axes[0, 1].boxplot(hfr_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[0, 1].set_title('High-Frequency Ratio Distribution', fontsize=12)
        axes[0, 1].set_ylabel('HFR')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 能量集中度
        conc_data = []
        for t in target_types:
            if metrics_by_type[t]['concentration']:
                conc_data.append(metrics_by_type[t]['concentration'])
        
        if conc_data:
            bp = axes[0, 2].boxplot(conc_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[0, 2].set_title('Energy Concentration Distribution', fontsize=12)
        axes[0, 2].set_ylabel('Concentration')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 平均指标对比
        metric_means = {t: {} for t in target_types}
        for t in target_types:
            if metrics_by_type[t]['entropy']:
                metric_means[t] = {
                    'entropy': np.mean(metrics_by_type[t]['entropy']),
                    'hfr': np.mean(metrics_by_type[t]['hfr']),
                    'concentration': np.mean(metrics_by_type[t]['concentration'])
                }
        
        x = np.arange(3)
        width = 0.25
        
        for i, t in enumerate(target_types):
            if t in metric_means and metric_means[t]:
                values = [metric_means[t]['entropy'], metric_means[t]['hfr'], metric_means[t]['concentration']]
                axes[1, 0].bar(x + i * width, values, width, label=t.value, color=self.colors[t], alpha=0.7)
        
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(['Entropy', 'HFR', 'Concentration'])
        axes[1, 0].set_title('Average Metrics Comparison', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. 正确 vs 错误预测对比
        correct_entropy = {t: [] for t in target_types}
        incorrect_entropy = {t: [] for t in target_types}
        
        for result in all_results:
            for seg_type, metrics_list in result.metrics_by_type.items():
                if seg_type in target_types:
                    for m in metrics_list:
                        if result.is_correct:
                            correct_entropy[seg_type].append(m.spectral_entropy)
                        else:
                            incorrect_entropy[seg_type].append(m.spectral_entropy)
        
        x = np.arange(len(target_types))
        correct_means = [np.mean(correct_entropy[t]) if correct_entropy[t] else 0 for t in target_types]
        incorrect_means = [np.mean(incorrect_entropy[t]) if incorrect_entropy[t] else 0 for t in target_types]
        
        axes[1, 1].bar(x - 0.2, correct_means, 0.4, label='Correct', color='green', alpha=0.7)
        axes[1, 1].bar(x + 0.2, incorrect_means, 0.4, label='Incorrect', color='red', alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([t.value for t in target_types])
        axes[1, 1].set_title('Entropy: Correct vs Incorrect', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Token 分布
        token_counts = {t: [] for t in target_types}
        for result in all_results:
            for seg in result.segments:
                if seg.type in target_types:
                    token_counts[seg.type].append(seg.token_count)
        
        token_data = [token_counts[t] for t in target_types if token_counts[t]]
        if token_data:
            bp = axes[1, 2].boxplot(token_data, labels=[t.value for t in target_types if token_counts[t]], patch_artist=True)
            for patch, color in zip(bp['boxes'], [self.colors[t] for t in target_types if token_counts[t]]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[1, 2].set_title('Token Count Distribution', fontsize=12)
        axes[1, 2].set_ylabel('Tokens')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()
    
    def plot_energy_decay_aggregate(
        self,
        all_results: List[TrajectoryResult],
        save_name: str = "energy_decay_aggregate.png"
    ):
        """绘制聚合能量衰减曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        
        for seg_type in target_types:
            all_psd = []
            
            for result in all_results:
                if seg_type in result.metrics_by_type:
                    for m in result.metrics_by_type[seg_type]:
                        if len(m.psd) > 4:
                            target_len = 64
                            x_old = np.linspace(0, 1, len(m.psd))
                            x_new = np.linspace(0, 1, target_len)
                            psd_interp = np.interp(x_new, x_old, m.psd)
                            psd_interp = psd_interp / (np.max(psd_interp) + 1e-10)
                            all_psd.append(psd_interp)
            
            if all_psd:
                mean_psd = np.mean(all_psd, axis=0)
                std_psd = np.std(all_psd, axis=0)
                frequencies = np.linspace(0, 1, len(mean_psd))
                
                color = self.colors[seg_type]
                ax.plot(frequencies, mean_psd, color=color, label=f'{seg_type.value} (n={len(all_psd)})', linewidth=2)
                ax.fill_between(frequencies, 
                               np.maximum(mean_psd - std_psd, 1e-6),
                               mean_psd + std_psd,
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Normalized Frequency', fontsize=12)
        ax.set_ylabel('Normalized PSD', fontsize=12)
        ax.set_title('Aggregated Energy Decay Curves', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xlim([0, 0.5])
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()
    
    def plot_trajectory_overview(
        self,
        result: TrajectoryResult,
        kv_cache: KVCacheData,
        save_name: str = None
    ):
        """绘制单条轨迹概览"""
        if save_name is None:
            save_name = f"trajectory_{result.trajectory_id}_overview.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. 频谱热力图
        keys = kv_cache.keys[0].numpy()
        num_heads, seq_len, head_dim = keys.shape
        keys_avg = np.mean(keys, axis=0)
        
        window_size = min(32, seq_len // 4)
        window_size = max(window_size, 4)
        hop_size = max(1, window_size // 4)
        
        spectrograms = []
        for i in range(0, seq_len - window_size + 1, hop_size):
            window = keys_avg[i:i+window_size]
            avg_signal = np.mean(window, axis=1)
            avg_signal = avg_signal - np.mean(avg_signal)
            std = np.std(avg_signal)
            if std > 1e-8:
                avg_signal = avg_signal / std
            spectrum = np.abs(dct(avg_signal, norm='ortho'))
            spectrograms.append(spectrum)
        
        if spectrograms:
            spectrogram = np.array(spectrograms).T
            im = axes[0, 0].imshow(
                np.log1p(spectrogram), 
                aspect='auto', 
                cmap='magma',
                extent=[0, seq_len, window_size//2, 0],
                interpolation='bilinear'
            )
            
            for seg in result.segments:
                if seg.type in [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]:
                    color = self.colors[seg.type]
                    axes[0, 0].axvline(x=seg.start_token_idx, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
            
            plt.colorbar(im, ax=axes[0, 0], label='Log Power')
        
        axes[0, 0].set_xlabel('Token Position')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Trajectory {result.trajectory_id} Spectrogram (Layer {kv_cache.layer_idx})')
        
        # 2. 片段指标
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        entropy_data = []
        labels = []
        colors = []
        
        for t in target_types:
            if t in result.metrics_by_type and result.metrics_by_type[t]:
                entropy_data.append([m.spectral_entropy for m in result.metrics_by_type[t]])
                labels.append(t.value)
                colors.append(self.colors[t])
        
        if entropy_data:
            bp = axes[0, 1].boxplot(entropy_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[0, 1].set_title('Spectral Entropy by Segment Type')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 片段 token 分布
        type_tokens = {}
        for seg in result.segments:
            t = seg.type
            type_tokens[t] = type_tokens.get(t, 0) + seg.token_count
        
        types = [t for t in target_types if t in type_tokens]
        tokens = [type_tokens[t] for t in types]
        colors_bar = [self.colors[t] for t in types]
        
        axes[1, 0].bar([t.value for t in types], tokens, color=colors_bar, alpha=0.7)
        axes[1, 0].set_title('Token Count by Segment Type')
        axes[1, 0].set_ylabel('Tokens')
        axes[1, 0].tick_params(axis='x', rotation=30)
        
        # 4. 信息面板
        axes[1, 1].axis('off')
        info_text = f"""
Trajectory ID: {result.trajectory_id}
Question: {result.question[:100]}{'...' if len(result.question) > 100 else ''}

Answer: {result.answer}
Prediction: {result.prediction}
Correct: {'✓' if result.is_correct else '✗'}

Total Tokens: {result.token_count}
Analysis Layer: {result.layer_idx}
        """
        axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class AgentTrajectoryBatchExperiment:
    """批量分析实验"""
    
    def __init__(
        self,
        model_path: str = "/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B",
        target_layers: Optional[List[int]] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        save_dir: str = "./spectral_results"
    ):
        print("=" * 70)
        print("  Agent 轨迹频谱批量分析实验")
        print("  基于 WebSailor-3B 真实 KV Cache")
        print("=" * 70)
        
        self.kv_extractor = WebSailorKVExtractor(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit
        )
        
        self.parser = AgentTrajectoryParser(self.kv_extractor.tokenizer)
        self.spectral_analyzer = KVCacheSpectralAnalyzer(cutoff_ratio=0.2)
        self.visualizer = BatchVisualizer(save_dir=save_dir)
        
        if target_layers is None:
            n = self.kv_extractor.num_layers
            self.target_layers = [n // 2, n * 3 // 4, n - 1]
        else:
            self.target_layers = target_layers
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"目标分析层: {self.target_layers}")
        print(f"结果保存目录: {self.save_dir}")
    
    def load_trajectories(self, json_path: str) -> List[dict]:
        """从 JSON 文件加载轨迹数据"""
        print(f"\n加载数据: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, dict) and 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            trajectories = [data]
        
        print(f"加载了 {len(trajectories)} 条轨迹")
        return trajectories
    
    def analyze_single_trajectory(
        self, 
        json_data: dict,
        trajectory_id: int,
        max_length: int = 8192,
        save_individual: bool = False
    ) -> TrajectoryResult:
        """分析单条轨迹"""
        
        # 解析轨迹
        full_text, segments = self.parser.parse_trajectory(json_data)
        
        # 提取 KV Cache
        kv_cache_dict, token_ids = self.kv_extractor.extract_kv_cache(
            full_text, 
            target_layers=self.target_layers,
            max_length=max_length
        )
        
        # 选择主分析层
        main_layer = self.target_layers[len(self.target_layers) // 2]
        main_kv = kv_cache_dict[main_layer]
        
        # 分析各片段
        metrics_by_type: Dict[SegmentType, List[SpectralMetrics]] = {t: [] for t in SegmentType}
        
        for seg in segments:
            if seg.token_count < 4:
                continue
            if seg.end_token_idx > main_kv.keys.shape[2]:
                continue
            
            keys, values = self.spectral_analyzer.extract_segment_kv(main_kv, seg)
            if keys.size == 0:
                continue
            
            metrics = self.spectral_analyzer.analyze_segment_kv(keys, values)
            metrics_by_type[seg.type].append(metrics)
        
        # 汇总
        summary = {}
        for seg_type, metrics_list in metrics_by_type.items():
            if metrics_list:
                summary[seg_type.value] = {
                    'count': len(metrics_list),
                    'spectral_entropy_mean': np.mean([m.spectral_entropy for m in metrics_list]),
                    'spectral_entropy_std': np.std([m.spectral_entropy for m in metrics_list]),
                    'high_freq_ratio_mean': np.mean([m.high_freq_ratio for m in metrics_list]),
                    'low_freq_ratio_mean': np.mean([m.low_freq_ratio for m in metrics_list]),
                    'energy_concentration_mean': np.mean([m.energy_concentration for m in metrics_list]),
                }
        
        # 获取元信息
        question = json_data.get('question', '')
        if not question:
            for msg in json_data.get('messages', []):
                if msg.get('role') == 'user' and '<tool_response>' not in msg.get('content', ''):
                    question = msg.get('content', '')[:200]
                    break
        
        answer = json_data.get('answer', '')
        prediction = json_data.get('prediction', '')
        is_correct = str(answer).strip() == str(prediction).strip() if answer and prediction else False
        
        result = TrajectoryResult(
            trajectory_id=trajectory_id,
            question=question,
            answer=answer,
            prediction=prediction,
            is_correct=is_correct,
            segments=segments,
            metrics_by_type=metrics_by_type,
            summary=summary,
            token_count=len(token_ids),
            layer_idx=main_layer
        )
        
        # 可选：保存单条轨迹可视化
        if save_individual:
            self.visualizer.plot_trajectory_overview(result, main_kv)
        
        return result
    
    def run_batch_analysis(
        self, 
        json_path: str,
        max_trajectories: Optional[int] = None,
        max_length: int = 8192,
        save_individual: bool = False
    ) -> List[TrajectoryResult]:
        """运行批量分析"""
        
        # 加载数据
        trajectories = self.load_trajectories(json_path)
        
        if max_trajectories:
            trajectories = trajectories[:max_trajectories]
        
        print(f"\n开始分析 {len(trajectories)} 条轨迹...")
        print("=" * 60)
        
        all_results = []
        
        for i, traj in enumerate(tqdm(trajectories, desc="分析轨迹")):
            try:
                result = self.analyze_single_trajectory(
                    traj,
                    trajectory_id=traj.get('rollout_id', i),
                    max_length=max_length,
                    save_individual=save_individual
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n警告: 轨迹 {i} 分析失败: {e}")
                continue
        
        print(f"\n成功分析 {len(all_results)} 条轨迹")
        
        return all_results
    
    def generate_report(self, all_results: List[TrajectoryResult]) -> Dict:
        """生成分析报告"""
        
        print("\n" + "=" * 70)
        print("  批量分析报告")
        print("=" * 70)
        
        # 统计
        total = len(all_results)
        correct = sum(1 for r in all_results if r.is_correct)
        
        print(f"\n总轨迹数: {total}")
        print(f"正确预测: {correct} ({100*correct/total:.1f}%)")
        
        # 聚合指标
        aggregate_metrics = {t: {'entropy': [], 'hfr': [], 'concentration': []} 
                           for t in SegmentType}
        
        for result in all_results:
            for seg_type, metrics_list in result.metrics_by_type.items():
                for m in metrics_list:
                    aggregate_metrics[seg_type]['entropy'].append(m.spectral_entropy)
                    aggregate_metrics[seg_type]['hfr'].append(m.high_freq_ratio)
                    aggregate_metrics[seg_type]['concentration'].append(m.energy_concentration)
        
        print("\n" + "-" * 50)
        print("聚合频谱指标:")
        print("-" * 50)
        
        target_types = [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]
        
        for seg_type in target_types:
            if aggregate_metrics[seg_type]['entropy']:
                n = len(aggregate_metrics[seg_type]['entropy'])
                se_mean = np.mean(aggregate_metrics[seg_type]['entropy'])
                se_std = np.std(aggregate_metrics[seg_type]['entropy'])
                hfr_mean = np.mean(aggregate_metrics[seg_type]['hfr'])
                conc_mean = np.mean(aggregate_metrics[seg_type]['concentration'])
                
                print(f"\n[{seg_type.value.upper()}] ({n} 个片段)")
                print(f"  谱熵:     {se_mean:.4f} ± {se_std:.4f}")
                print(f"  HFR:      {hfr_mean:.4f}")
                print(f"  能量集中: {conc_mean:.4f}")
        
        # 假设验证
        print("\n" + "-" * 50)
        print("假设验证:")
        print("-" * 50)
        
        if aggregate_metrics[SegmentType.THINK]['entropy'] and aggregate_metrics[SegmentType.TOOL_CALL]['entropy']:
            think_se = np.mean(aggregate_metrics[SegmentType.THINK]['entropy'])
            call_se = np.mean(aggregate_metrics[SegmentType.TOOL_CALL]['entropy'])
            think_hfr = np.mean(aggregate_metrics[SegmentType.THINK]['hfr'])
            call_hfr = np.mean(aggregate_metrics[SegmentType.TOOL_CALL]['hfr'])
            
            print(f"\n1. 谱熵对比:")
            print(f"   Think: {think_se:.4f}  vs  Tool Call: {call_se:.4f}")
            if think_se > call_se:
                print("   ✓ Think 谱熵更高 → 信号更接近噪声，存在压缩空间")
            else:
                print("   ✗ 假设不成立")
            
            print(f"\n2. 高频能量占比对比:")
            print(f"   Think: {think_hfr:.4f}  vs  Tool Call: {call_hfr:.4f}")
            if think_hfr > call_hfr:
                print("   ✓ Think 高频成分更多 → 可通过低通滤波压缩")
            else:
                print("   ✗ 假设不成立")
        
        # 正确 vs 错误对比
        print("\n" + "-" * 50)
        print("正确 vs 错误预测对比:")
        print("-" * 50)
        
        correct_entropy = {t: [] for t in target_types}
        incorrect_entropy = {t: [] for t in target_types}
        
        for result in all_results:
            for seg_type in target_types:
                if seg_type in result.metrics_by_type:
                    for m in result.metrics_by_type[seg_type]:
                        if result.is_correct:
                            correct_entropy[seg_type].append(m.spectral_entropy)
                        else:
                            incorrect_entropy[seg_type].append(m.spectral_entropy)
        
        for seg_type in target_types:
            if correct_entropy[seg_type] and incorrect_entropy[seg_type]:
                c_mean = np.mean(correct_entropy[seg_type])
                i_mean = np.mean(incorrect_entropy[seg_type])
                print(f"\n{seg_type.value}:")
                print(f"  正确预测谱熵: {c_mean:.4f}")
                print(f"  错误预测谱熵: {i_mean:.4f}")
                if c_mean < i_mean:
                    print("  → 正确预测的谱熵更低（信号更结构化）")
        
        # 返回汇总数据
        report_data = {
            'total_trajectories': total,
            'correct_predictions': correct,
            'accuracy': correct / total if total > 0 else 0,
            'aggregate_metrics': {
                seg_type.value: {
                    'count': len(aggregate_metrics[seg_type]['entropy']),
                    'entropy_mean': np.mean(aggregate_metrics[seg_type]['entropy']) if aggregate_metrics[seg_type]['entropy'] else 0,
                    'entropy_std': np.std(aggregate_metrics[seg_type]['entropy']) if aggregate_metrics[seg_type]['entropy'] else 0,
                    'hfr_mean': np.mean(aggregate_metrics[seg_type]['hfr']) if aggregate_metrics[seg_type]['hfr'] else 0,
                    'concentration_mean': np.mean(aggregate_metrics[seg_type]['concentration']) if aggregate_metrics[seg_type]['concentration'] else 0,
                }
                for seg_type in target_types
            }
        }
        
        return report_data
    
    def visualize_all(self, all_results: List[TrajectoryResult]):
        """生成所有可视化"""
        print("\n生成可视化图表...")
        
        self.visualizer.plot_aggregate_statistics(all_results)
        self.visualizer.plot_energy_decay_aggregate(all_results)
        
        print(f"所有图表已保存到: {self.save_dir}")
    
    def save_results(self, all_results: List[TrajectoryResult], report_data: Dict):
        """保存结果到 JSON"""
        
        # 转换为可序列化格式
        results_data = []
        for r in all_results:
            results_data.append({
                'trajectory_id': r.trajectory_id,
                'question': r.question[:200],
                'answer': r.answer,
                'prediction': r.prediction,
                'is_correct': r.is_correct,
                'token_count': r.token_count,
                'layer_idx': r.layer_idx,
                'summary': r.summary
            })
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'report': report_data,
            'trajectories': results_data
        }
        
        output_path = self.save_dir / 'analysis_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent 轨迹频谱批量分析')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default='./spectral_results', help='输出目录')
    parser.add_argument('--model', '-m', type=str, 
                       default='/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B',
                       help='模型路径')
    parser.add_argument('--max-trajectories', '-n', type=int, default=None, help='最大分析轨迹数')
    parser.add_argument('--max-length', type=int, default=8192, help='最大序列长度')
    parser.add_argument('--save-individual', action='store_true', help='保存单条轨迹可视化')
    parser.add_argument('--load-in-4bit', action='store_true', help='使用 4bit 量化')
    
    args = parser.parse_args()
    
    # 创建实验
    experiment = AgentTrajectoryBatchExperiment(
        model_path=args.model,
        save_dir=args.output,
        load_in_4bit=args.load_in_4bit
    )
    
    # 运行分析
    all_results = experiment.run_batch_analysis(
        json_path=args.input,
        max_trajectories=args.max_trajectories,
        max_length=args.max_length,
        save_individual=args.save_individual
    )
    
    # 生成报告
    report_data = experiment.generate_report(all_results)
    
    # 生成可视化
    experiment.visualize_all(all_results)
    
    # 保存结果
    experiment.save_results(all_results, report_data)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()